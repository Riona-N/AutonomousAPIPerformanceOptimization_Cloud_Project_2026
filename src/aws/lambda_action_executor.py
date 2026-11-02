"""Action Executor Lambda Function (Role 2).

Executes routing decisions produced by the RL agent:
1. Coordinates transaction execution across the 5 microservice stages using the chosen candidates.
2. Measures realized end-to-end latency, checks SLA violation and timeout thresholds.
3. Computes the realized multi-objective reward and updates the audit explanation.
4. Writes an immutable audit record to the DynamoDB TransactionLogs table.
5. Emits the execution payload for downstream metric publishing.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

try:
    from src.aws.dynamodb_helpers import log_transaction
    from src.aws.lambda_microservices import microservice_pipeline_handler
except ImportError:
    from dynamodb_helpers import log_transaction
    from lambda_microservices import microservice_pipeline_handler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_SLA_MS = 374.19
DEFAULT_TIMEOUT_MS = 748.38


def compute_realized_reward(
    e2e_ms: float,
    sla_ms: float,
    timeout_ms: float,
    fraud_risk: float,
) -> tuple[float, dict[str, Any]]:
    """Compute the ground-truth realized reward matching Riona's RL contract."""
    success = e2e_ms <= timeout_ms
    sla_violation = e2e_ms > sla_ms

    success_bonus = 1.0 if success else -1.0
    latency_penalty = -e2e_ms / sla_ms
    sla_penalty = -0.75 if sla_violation else 0.0
    fraud_bonus = 0.20 * (1.0 - fraud_risk)

    reward = success_bonus + latency_penalty + sla_penalty + fraud_bonus

    terms = {
        "success_bonus": round(success_bonus, 4),
        "latency_penalty": round(latency_penalty, 4),
        "sla_violation_penalty": round(sla_penalty, 4),
        "fraud_risk_bonus": round(fraud_bonus, 4),
    }
    dominant = max(terms, key=lambda k: abs(terms[k]))

    explanation = {
        "terms": terms,
        "dominant_reward_component": dominant,
        "dominant_value": terms[dominant],
        "e2e_ms": round(e2e_ms, 2),
        "estimated": False,
    }
    return round(reward, 4), explanation


def lambda_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Action Executor entry point."""
    logger.info(f"Action Executor processing transaction: {event.get('transaction_id')}")

    tx_id = event.get("transaction_id", f"txn_{int(time.time() * 1000)}")
    tx = event.get("transaction", {})
    action = event.get("action", {})
    sla_ms = float(event.get("sla_ms", DEFAULT_SLA_MS))
    timeout_ms = float(event.get("timeout_ms", DEFAULT_TIMEOUT_MS))
    fraud_risk = float(tx.get("fraud_risk", 0.01))

    # 1. Execute microservice pipeline
    pipeline_result = microservice_pipeline_handler(event, context)
    e2e_ms = float(pipeline_result["e2e_ms"])
    stage_latencies = pipeline_result["stage_latencies_ms"]
    success = pipeline_result["success"]
    sla_violation = pipeline_result["sla_violation"]

    # 2. Compute realized reward & audit explanation
    reward, realized_explanation = compute_realized_reward(
        e2e_ms=e2e_ms,
        sla_ms=sla_ms,
        timeout_ms=timeout_ms,
        fraud_risk=fraud_risk,
    )

    # 3. Log to DynamoDB for financial compliance audit
    log_record = {
        "transaction_id": tx_id,
        "timestamp": int(time.time() * 1000),
        "type": tx.get("type", "PAYMENT"),
        "amount": tx.get("amount", 0.0),
        "fraud_risk": fraud_risk,
        "routing_action": action,
        "stage_latencies_ms": stage_latencies,
        "e2e_ms": e2e_ms,
        "sla_violation": int(sla_violation),
        "success": int(success),
        "reward": reward,
        "dominant_reward_component": realized_explanation["dominant_reward_component"],
        "explanation": json.dumps(realized_explanation),
    }

    try:
        log_transaction(log_record)
        logger.info(f"Transaction {tx_id} logged to DynamoDB successfully.")
    except Exception as e:
        logger.warning(f"Could not log transaction to DynamoDB ({e}); continuing.")

    response = {
        "statusCode": 200,
        "transaction_id": tx_id,
        "transaction": tx,
        "action": action,
        "stage_latencies_ms": stage_latencies,
        "e2e_ms": e2e_ms,
        "success": success,
        "sla_violation": sla_violation,
        "reward": reward,
        "explanation": realized_explanation,
        "sla_ms": sla_ms,
        "timeout_ms": timeout_ms,
    }
    return response


if __name__ == "__main__":
    dummy_event = {
        "transaction_id": "tx_demo_001",
        "transaction": {"type": "PAYMENT", "amount": 500.0, "fraud_risk": 0.02},
        "action": {"ingress": 0, "auth": 1, "routing": 0, "processor": 2, "db": 1},
        "sla_ms": 374.19,
        "timeout_ms": 748.38,
    }
    res = lambda_handler(dummy_event)
    print(f"Action Executor executed with e2e_ms={res['e2e_ms']}, success={res['success']}, reward={res['reward']}")
