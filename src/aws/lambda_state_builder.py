"""State Builder Lambda Function (Role 2).

Constructs the 83-dimensional observation vector required by the RL agent:
- 75 features: 5 stages × 3 candidates × 5 metrics (p50, p95, p99, log1p(n_calls), sla_flag)
- 5 features: one-hot transaction type flags
- 3 features: normalized log amount, fraud risk score, transfer flag

Queries live system metrics from DynamoDB / CloudWatch (or local baseline fallback),
invokes the SageMaker endpoint (or local policy fallback), and passes the state + action
down the pipeline.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

try:
    import numpy as np
except ImportError:
    np = None

try:
    from src.aws.dynamodb_helpers import get_metrics_cache
except ImportError:
    from dynamodb_helpers import get_metrics_cache

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

STAGES = ("ingress", "auth", "routing", "processor", "db")
CANDIDATES_PER_STAGE = 3
METRICS_PER_CANDIDATE = 5
TX_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")

DEFAULT_SLA_MS = 374.19
DEFAULT_TIMEOUT_MS = 748.38
DEFAULT_AMOUNT_SCALE = 1_000_000.0


def _clip(val: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(val)))


def _load_baseline_metrics_from_disk() -> dict[str, dict[str, float]]:
    """Fallback to processed parquet metrics if cache/CloudWatch is cold."""
    try:
        import pandas as pd

        possible_paths = [
            Path("dataset/processed/stage_latency_windows.parquet"),
            Path(__file__).resolve().parents[2] / "dataset" / "processed" / "stage_latency_windows.parquet",
        ]
        target = next((p for p in possible_paths if p.exists()), None)
        if target:
            df = pd.read_parquet(target)
            first_window = df["window_id"].min()
            window_df = df[df["window_id"] == first_window]
            cache = {}
            for row in window_df.itertuples():
                key = f"{row.stage}#{row.cand_rank}"
                cache[key] = {
                    "stage": row.stage,
                    "candidate": int(row.cand_rank),
                    "p50": float(row.p50),
                    "p95": float(row.p95),
                    "p99": float(row.p99),
                    "n_calls": int(row.n_calls),
                    "sla_violation": int(row.sla_violation),
                }
            return cache
    except Exception as e:
        logger.warning(f"Could not load fallback parquet: {e}")

    # Built-in synthetic fallback if parquet not found
    fallback = {}
    for stage in STAGES:
        for cand in range(CANDIDATES_PER_STAGE):
            fallback[f"{stage}#{cand}"] = {
                "stage": stage,
                "candidate": cand,
                "p50": 30.0 + cand * 5.0,
                "p95": 60.0 + cand * 10.0,
                "p99": 100.0 + cand * 20.0,
                "n_calls": 100,
                "sla_violation": 0,
            }
    return fallback


def build_observation_vector(
    tx_event: dict[str, Any],
    metrics_cache: dict[str, dict[str, Any]],
    timeout_ms: float = DEFAULT_TIMEOUT_MS,
    amount_scale: float = DEFAULT_AMOUNT_SCALE,
) -> np.ndarray:
    """Build the normalized 83-dimensional observation vector."""
    # 1. 75 metric features (5 stages x 3 candidates x 5 metrics)
    metric_values = []
    for stage in STAGES:
        for cand in range(CANDIDATES_PER_STAGE):
            key = f"{stage}#{cand}"
            m = metrics_cache.get(key, {})
            p50 = float(m.get("p50", 40.0))
            p95 = float(m.get("p95", 75.0))
            p99 = float(m.get("p99", 120.0))
            n_calls = float(m.get("n_calls", 50))
            sla_viol = float(m.get("sla_violation", 0))

            metric_values.extend([
                _clip(p50 / timeout_ms, 0.0, 1.0),
                _clip(p95 / timeout_ms, 0.0, 1.0),
                _clip(p99 / timeout_ms, 0.0, 1.0),
                _clip(math.log1p(n_calls) / 10.0, 0.0, 1.0),
                float(1.0 if sla_viol > 0 else 0.0),
            ])

    # 2. 5 one-hot type features
    tx_type = str(tx_event.get("type", "PAYMENT")).upper()
    type_values = [1.0 if tx_type == t else 0.0 for t in TX_TYPES]

    # 3. 3 transaction features
    amount = float(tx_event.get("amount", 100.0))
    norm_amount = _clip(math.log1p(amount) / math.log1p(amount_scale), 0.0, 1.0)
    fraud_risk = _clip(float(tx_event.get("fraud_risk", 0.01)), 0.0, 1.0)
    is_transfer = 1.0 if tx_type.startswith("TRANSFER") else 0.0

    tx_values = [norm_amount, fraud_risk, is_transfer]

    full_values = metric_values + type_values + tx_values
    if np is not None:
        obs = np.asarray(full_values, dtype=np.float32)
    else:
        obs = full_values
    assert len(obs) == 83, f"Expected 83 features, got {len(obs)}"
    return obs


def invoke_routing_agent(
    observation: np.ndarray,
    context: dict[str, Any],
    endpoint_name: str | None = None,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Invoke SageMaker endpoint or perform robust local fallback."""
    endpoint_name = endpoint_name or os.getenv("SAGEMAKER_ENDPOINT_NAME")
    region = os.getenv("AWS_REGION", "us-east-1")

    if endpoint_name:
        try:
            client = boto3.client("sagemaker-runtime", region_name=region)
            payload = {
                "instances": [{
                    "observation": observation.tolist(),
                    "context": context,
                }]
            }
            response = client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload),
            )
            result = json.loads(response["Body"].read().decode("utf-8"))
            pred = result["predictions"][0]
            return pred["action"], pred["explanation"]
        except Exception as exc:
            logger.warning(f"SageMaker invocation failed ({exc}); falling back to local policy.")

    # Local Fallback Policy:
    # Greedily pick candidate with lowest p95 latency from observation
    action = {}
    total_est_latency = 0.0
    timeout_ms = float(context.get("timeout_ms", DEFAULT_TIMEOUT_MS))
    sla_ms = float(context.get("sla_ms", DEFAULT_SLA_MS))
    fraud_risk = float(context.get("fraud_risk", observation[-2]))

    for stage_idx, stage in enumerate(STAGES):
        best_cand = 0
        lowest_p95 = float("inf")
        for cand in range(CANDIDATES_PER_STAGE):
            p95_feat = observation[stage_idx * 15 + cand * METRICS_PER_CANDIDATE + 1]
            if p95_feat < lowest_p95:
                lowest_p95 = p95_feat
                best_cand = cand
        action[stage] = int(best_cand)
        total_est_latency += lowest_p95 * timeout_ms

    terms = {
        "success_bonus": 1.0 if total_est_latency <= timeout_ms else -1.0,
        "latency_penalty": -total_est_latency / sla_ms,
        "sla_violation_penalty": -0.75 if total_est_latency > sla_ms else 0.0,
        "fraud_risk_bonus": 0.20 * (1.0 - fraud_risk),
    }
    dominant = max(terms, key=lambda k: abs(terms[k]))
    explanation = {
        "terms": terms,
        "dominant_reward_component": dominant,
        "dominant_value": terms[dominant],
        "e2e_ms": total_est_latency,
        "estimated": True,
    }
    return action, explanation


def lambda_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Lambda entry point for State Builder."""
    logger.info(f"Received event: {json.dumps(event)}")

    # Normalize body if event is from API Gateway
    if "body" in event and isinstance(event["body"], str):
        try:
            body = json.loads(event["body"])
        except json.JSONDecodeError:
            body = event
    else:
        body = event.get("body", event)

    tx_id = str(body.get("transaction_id", f"txn_{random.randint(1_000_000, 9_000_000)}"))
    tx_event = {
        "transaction_id": tx_id,
        "type": body.get("type", "PAYMENT"),
        "amount": float(body.get("amount", 250.0)),
        "fraud_risk": float(body.get("fraud_risk", 0.01)),
        "slot": int(body.get("slot", 0)),
        "window_id": int(body.get("window_id", 0)),
    }

    sla_ms = float(body.get("sla_ms", os.getenv("SLA_MS", DEFAULT_SLA_MS)))
    timeout_ms = float(body.get("timeout_ms", os.getenv("TIMEOUT_MS", DEFAULT_TIMEOUT_MS)))

    # Fetch live metrics from DynamoDB cache
    metrics_cache = get_metrics_cache()
    if not metrics_cache:
        metrics_cache = _load_baseline_metrics_from_disk()

    # Build 83-dim observation
    observation = build_observation_vector(
        tx_event=tx_event,
        metrics_cache=metrics_cache,
        timeout_ms=timeout_ms,
    )

    # Invoke RL agent / SageMaker
    ctx = {
        "sla_ms": sla_ms,
        "timeout_ms": timeout_ms,
        "fraud_risk": tx_event["fraud_risk"],
    }
    action, explanation = invoke_routing_agent(observation, ctx)

    result = {
        "statusCode": 200,
        "transaction_id": tx_id,
        "transaction": tx_event,
        "observation": observation.tolist(),
        "action": action,
        "explanation": explanation,
        "sla_ms": sla_ms,
        "timeout_ms": timeout_ms,
    }
    logger.info(f"State builder generated action: {action}")
    return result


if __name__ == "__main__":
    test_event = {"type": "TRANSFER", "amount": 15000.0, "fraud_risk": 0.05}
    out = lambda_handler(test_event)
    print(f"Observation size: {len(out['observation'])}")
    print(f"Action: {out['action']}")
    print(f"Explanation: {out['explanation']}")
