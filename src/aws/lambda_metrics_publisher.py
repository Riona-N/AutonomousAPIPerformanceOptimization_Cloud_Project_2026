"""Metrics Publisher Lambda Function (Role 2).

Closes the feedback loop by:
1. Publishing custom CloudWatch metrics for latency, SLA compliance, success rate, and RL reward.
2. Updating the DynamoDB SystemMetricsCache so subsequent state-building cycles observe the
   consequences of recent routing decisions.
3. Returning the final HTTP / Step Functions response to the caller.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError

try:
    from src.aws.dynamodb_helpers import update_metrics_cache
except ImportError:
    from dynamodb_helpers import update_metrics_cache

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

METRICS_NAMESPACE = os.getenv("CLOUDWATCH_NAMESPACE", "AutonomousPaymentAPI")


def publish_cloudwatch_metrics(
    payload: dict[str, Any],
    cloudwatch_client: Any = None,
) -> bool:
    """Publish custom metrics to AWS CloudWatch."""
    if cloudwatch_client is None:
        region = os.getenv("AWS_REGION", "us-east-1")
        cloudwatch_client = boto3.client("cloudwatch", region_name=region)

    e2e_ms = float(payload.get("e2e_ms", 0.0))
    sla_violation = 1.0 if payload.get("sla_violation") else 0.0
    success = 1.0 if payload.get("success") else 0.0
    reward = float(payload.get("reward", 0.0))
    fraud_risk = float(payload.get("transaction", {}).get("fraud_risk", 0.0))

    dimensions = [
        {"Name": "System", "Value": "PaymentGateway"},
        {"Name": "Policy", "Value": "Autonomous_RL"},
    ]

    metric_data = [
        {
            "MetricName": "EndToEndLatency",
            "Dimensions": dimensions,
            "Value": e2e_ms,
            "Unit": "Milliseconds",
        },
        {
            "MetricName": "SLAViolation",
            "Dimensions": dimensions,
            "Value": sla_violation,
            "Unit": "Count",
        },
        {
            "MetricName": "TransactionSuccess",
            "Dimensions": dimensions,
            "Value": success,
            "Unit": "Count",
        },
        {
            "MetricName": "RLReward",
            "Dimensions": dimensions,
            "Value": reward,
            "Unit": "None",
        },
        {
            "MetricName": "FraudRiskScore",
            "Dimensions": dimensions,
            "Value": fraud_risk,
            "Unit": "None",
        },
    ]

    # Add per-stage latency metrics
    stage_latencies = payload.get("stage_latencies_ms", {})
    for stage, lat in stage_latencies.items():
        metric_data.append({
            "MetricName": f"{stage.capitalize()}Latency",
            "Dimensions": dimensions,
            "Value": float(lat),
            "Unit": "Milliseconds",
        })

    try:
        cloudwatch_client.put_metric_data(
            Namespace=METRICS_NAMESPACE,
            MetricData=metric_data,
        )
        logger.info(f"Published {len(metric_data)} metrics to CloudWatch namespace {METRICS_NAMESPACE}")
        return True
    except Exception as exc:
        logger.warning(f"Could not publish metrics to CloudWatch ({exc}); continuing.")
        return False


def update_live_metrics_cache(payload: dict[str, Any]) -> int:
    """Update DynamoDB cache with latest observed performance to close the RL loop."""
    action = payload.get("action", {})
    stage_latencies = payload.get("stage_latencies_ms", {})
    sla_ms = float(payload.get("sla_ms", 374.19))

    cache_entries = []
    for stage, cand in action.items():
        lat = float(stage_latencies.get(stage, 50.0))
        cache_entries.append({
            "stage": stage,
            "candidate": int(cand),
            "p50": lat * 0.8,
            "p95": lat * 1.2,
            "p99": lat * 1.5,
            "n_calls": 100,
            "sla_violation": 1 if lat > (sla_ms / 5.0) else 0,
        })

    try:
        return update_metrics_cache(cache_entries)
    except Exception as exc:
        logger.warning(f"Could not update metrics cache ({exc}); continuing.")
        return 0


def lambda_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Metrics Publisher entry point."""
    logger.info(f"Publishing metrics for transaction: {event.get('transaction_id')}")

    # 1. Emit metrics to CloudWatch
    cw_success = publish_cloudwatch_metrics(event)

    # 2. Update DynamoDB system metrics cache (closed loop)
    cached_count = update_live_metrics_cache(event)

    response_body = {
        "status": "SUCCESS" if event.get("success") else "FAILED",
        "transaction_id": event.get("transaction_id"),
        "e2e_latency_ms": event.get("e2e_ms"),
        "sla_violation": event.get("sla_violation"),
        "reward": event.get("reward"),
        "routing_action": event.get("action"),
        "stage_latencies_ms": event.get("stage_latencies_ms"),
        "explanation": event.get("explanation"),
        "metrics_published_cloudwatch": cw_success,
        "metrics_cache_updated": cached_count,
    }

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(response_body),
    }


if __name__ == "__main__":
    dummy_payload = {
        "transaction_id": "tx_demo_002",
        "e2e_ms": 142.5,
        "success": True,
        "sla_violation": False,
        "reward": 1.25,
        "action": {"ingress": 0, "auth": 1, "routing": 0, "processor": 2, "db": 1},
        "stage_latencies_ms": {"ingress": 12.0, "auth": 24.5, "routing": 14.0, "processor": 55.0, "db": 37.0},
        "explanation": {"dominant_reward_component": "success_bonus"},
    }
    out = lambda_handler(dummy_payload)
    print("Response:", out)
