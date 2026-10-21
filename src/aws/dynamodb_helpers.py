"""DynamoDB helper utilities for transaction logging and metric state caching (Role 2)."""

from __future__ import annotations

import decimal
import json
import logging
import os
import time
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

TRANSACTION_LOGS_TABLE = os.getenv("DYNAMODB_TRANSACTION_TABLE", "TransactionLogs")
METRICS_CACHE_TABLE = os.getenv("DYNAMODB_METRICS_TABLE", "SystemMetricsCache")


def _get_resource(region_name: str | None = None) -> Any:
    try:
        return boto3.resource("dynamodb", region_name=region_name or os.getenv("AWS_REGION", "us-east-1"))
    except Exception as exc:
        logger.debug(f"Could not initialize DynamoDB resource: {exc}")
        return None


def float_to_decimal(obj: Any) -> Any:
    """Recursively convert float values to Decimal for DynamoDB serialization."""
    if isinstance(obj, float):
        return decimal.Decimal(str(round(obj, 6)))
    if isinstance(obj, dict):
        return {k: float_to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [float_to_decimal(v) for v in obj]
    return obj


def decimal_to_float(obj: Any) -> Any:
    """Recursively convert Decimal values back to float/int."""
    if isinstance(obj, decimal.Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    if isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decimal_to_float(v) for v in obj]
    return obj


def create_tables(dynamodb: Any = None) -> dict[str, str]:
    """Create DynamoDB tables if they do not exist."""
    if dynamodb is None:
        dynamodb = _get_resource()

    created = {}

    # 1. TransactionLogs table
    try:
        table = dynamodb.create_table(
            TableName=TRANSACTION_LOGS_TABLE,
            KeySchema=[
                {"AttributeName": "transaction_id", "KeyType": "HASH"},
                {"AttributeName": "timestamp", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "transaction_id", "AttributeType": "S"},
                {"AttributeName": "timestamp", "AttributeType": "N"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        table.wait_until_exists()
        created["transaction_logs"] = TRANSACTION_LOGS_TABLE
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            created["transaction_logs"] = f"{TRANSACTION_LOGS_TABLE} (already exists)"
        else:
            raise

    # 2. SystemMetricsCache table
    try:
        table = dynamodb.create_table(
            TableName=METRICS_CACHE_TABLE,
            KeySchema=[
                {"AttributeName": "stage_candidate", "KeyType": "HASH"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "stage_candidate", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        table.wait_until_exists()
        created["metrics_cache"] = METRICS_CACHE_TABLE
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            created["metrics_cache"] = f"{METRICS_CACHE_TABLE} (already exists)"
        else:
            raise

    return created


def log_transaction(
    item: dict[str, Any],
    table_name: str = TRANSACTION_LOGS_TABLE,
    dynamodb: Any = None,
) -> dict[str, Any]:
    """Write an execution record to the TransactionLogs DynamoDB table."""
    record = dict(item)
    if "timestamp" not in record:
        record["timestamp"] = int(time.time() * 1000)

    try:
        if dynamodb is None:
            dynamodb = _get_resource()
        if dynamodb is not None:
            table = dynamodb.Table(table_name)
            dynamo_item = float_to_decimal(record)
            table.put_item(Item=dynamo_item)
    except (ClientError, BotoCoreError, Exception) as exc:
        logger.warning(f"Could not log transaction to DynamoDB: {exc}")
    return record


def query_transaction(
    transaction_id: str,
    table_name: str = TRANSACTION_LOGS_TABLE,
    dynamodb: Any = None,
) -> list[dict[str, Any]]:
    """Retrieve all execution logs for a given transaction_id."""
    try:
        if dynamodb is None:
            dynamodb = _get_resource()
        if dynamodb is None:
            return []
        table = dynamodb.Table(table_name)
        from boto3.dynamodb.conditions import Key

        response = table.query(
            KeyConditionExpression=Key("transaction_id").eq(str(transaction_id))
        )
        items = response.get("Items", [])
        return [decimal_to_float(item) for item in items]
    except (ClientError, BotoCoreError, Exception) as exc:
        logger.warning(f"Could not query DynamoDB: {exc}")
        return []


def update_metrics_cache(
    metrics_list: list[dict[str, Any]],
    table_name: str = METRICS_CACHE_TABLE,
    dynamodb: Any = None,
) -> int:
    """Store the latest stage-candidate metrics into the SystemMetricsCache table."""
    now = int(time.time() * 1000)
    count = 0
    try:
        if dynamodb is None:
            dynamodb = _get_resource()
        if dynamodb is None:
            return 0
        table = dynamodb.Table(table_name)
        with table.batch_writer() as batch:
            for m in metrics_list:
                key = f"{m['stage']}#{m['candidate']}"
                entry = {
                    "stage_candidate": key,
                    "stage": m["stage"],
                    "candidate": int(m["candidate"]),
                    "p50": m.get("p50", 0.0),
                    "p95": m.get("p95", 0.0),
                    "p99": m.get("p99", 0.0),
                    "n_calls": m.get("n_calls", 0),
                    "sla_violation": int(m.get("sla_violation", 0)),
                    "updated_at": now,
                }
                batch.put_item(Item=float_to_decimal(entry))
                count += 1
    except (ClientError, BotoCoreError, Exception) as exc:
        logger.warning(f"Could not update metrics cache: {exc}")
    return count


def get_metrics_cache(
    table_name: str = METRICS_CACHE_TABLE,
    dynamodb: Any = None,
) -> dict[str, dict[str, Any]]:
    """Scan all cached stage-candidate metrics for constructing observation states."""
    try:
        if dynamodb is None:
            dynamodb = _get_resource()
        if dynamodb is None:
            return {}
        table = dynamodb.Table(table_name)
        response = table.scan()
        items = response.get("Items", [])
        result = {}
        for item in items:
            clean = decimal_to_float(item)
            result[clean["stage_candidate"]] = clean
        return result
    except (ClientError, BotoCoreError, Exception) as exc:
        logger.debug(f"Could not scan metrics cache table {table_name}: {exc}")
        return {}
