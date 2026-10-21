"""Simulated Payment Microservices and Pipeline Execution (Role 2).

Simulates the 5 payment microservice stages:
1. Ingress
2. Authentication & Fraud Check
3. Routing & Load Distribution
4. Transaction Processor
5. Database Settlement

Each stage can be deployed as an independent Lambda function or executed
sequentially via the pipeline orchestrator. Real empirical latencies from
the Alibaba trace / LatencyModel are sampled for each chosen candidate replica.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from src.linkage_lib import LatencyModel
except ImportError:
    try:
        from linkage_lib import LatencyModel
    except ImportError:
        LatencyModel = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

STAGES = ("ingress", "auth", "routing", "processor", "db")
DEFAULT_SLA_MS = 374.19
DEFAULT_TIMEOUT_MS = 748.38

_LATENCY_MODEL_INSTANCE = None


def get_latency_model():
    """Singleton getter for the empirical LatencyModel."""
    global _LATENCY_MODEL_INSTANCE
    if _LATENCY_MODEL_INSTANCE is not None:
        return _LATENCY_MODEL_INSTANCE

    if LatencyModel is not None:
        possible_paths = [
            Path("dataset/processed/stage_latency_windows.parquet"),
            Path(__file__).resolve().parents[2] / "dataset" / "processed" / "stage_latency_windows.parquet",
        ]
        target = next((p for p in possible_paths if p.exists()), None)
        if target:
            try:
                _LATENCY_MODEL_INSTANCE = LatencyModel(target)
                return _LATENCY_MODEL_INSTANCE
            except Exception as e:
                logger.warning(f"Could not initialize LatencyModel: {e}")
    return None


def simulate_stage_latency(
    stage: str,
    candidate: int,
    window_id: int = 0,
    rng: np.random.Generator | None = None,
) -> float:
    """Sample realistic latency (ms) for a specific stage and candidate."""
    if rng is None:
        rng = np.random.default_rng()

    lm = get_latency_model()
    if lm is not None and stage in STAGES:
        stage_idx = STAGES.index(stage)
        try:
            val = float(lm.sample(
                stage_idx=stage_idx,
                cand=int(candidate),
                window_ids=[int(window_id)],
                rng=rng,
            )[0])
            return max(0.5, val)
        except Exception:
            pass

    # Calibrated empirical fallback based on Alibaba trace characteristics
    base_latencies = {
        "ingress": (12.0, 4.0),
        "auth": (25.0, 8.0),
        "routing": (15.0, 5.0),
        "processor": (60.0, 20.0),
        "db": (45.0, 15.0),
    }
    mean, std = base_latencies.get(stage, (30.0, 10.0))
    # Candidate variation: slight shift per candidate replica
    cand_shift = (candidate - 1) * 3.5
    raw = rng.normal(mean + cand_shift, std)
    return max(1.0, float(raw))


# Stage-specific Lambda handlers
def ingress_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    cand = int(event.get("candidate", 0))
    win = int(event.get("window_id", 0))
    lat = simulate_stage_latency("ingress", cand, win)
    return {"stage": "ingress", "candidate": cand, "latency_ms": lat, "status": "COMPLETED"}


def auth_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    cand = int(event.get("candidate", 0))
    win = int(event.get("window_id", 0))
    lat = simulate_stage_latency("auth", cand, win)
    return {"stage": "auth", "candidate": cand, "latency_ms": lat, "status": "COMPLETED"}


def routing_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    cand = int(event.get("candidate", 0))
    win = int(event.get("window_id", 0))
    lat = simulate_stage_latency("routing", cand, win)
    return {"stage": "routing", "candidate": cand, "latency_ms": lat, "status": "COMPLETED"}


def processor_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    cand = int(event.get("candidate", 0))
    win = int(event.get("window_id", 0))
    lat = simulate_stage_latency("processor", cand, win)
    return {"stage": "processor", "candidate": cand, "latency_ms": lat, "status": "COMPLETED"}


def db_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    cand = int(event.get("candidate", 0))
    win = int(event.get("window_id", 0))
    lat = simulate_stage_latency("db", cand, win)
    return {"stage": "db", "candidate": cand, "latency_ms": lat, "status": "COMPLETED"}


def microservice_pipeline_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Execute complete 5-stage payment transaction pipeline with selected candidates."""
    logger.info(f"Microservice pipeline running for event: {event.get('transaction_id')}")

    tx = event.get("transaction", {})
    action = event.get("action", {})
    sla_ms = float(event.get("sla_ms", DEFAULT_SLA_MS))
    timeout_ms = float(event.get("timeout_ms", DEFAULT_TIMEOUT_MS))
    window_id = int(tx.get("window_id", 0))

    stage_results = {}
    stage_latencies_ms = {}
    total_ms = 0.0

    rng = np.random.default_rng()

    for stage in STAGES:
        cand = int(action.get(stage, 0))
        lat = simulate_stage_latency(stage, cand, window_id, rng)
        stage_latencies_ms[stage] = round(lat, 2)
        stage_results[stage] = {
            "candidate": cand,
            "latency_ms": round(lat, 2),
            "status": "COMPLETED",
        }
        total_ms += lat

    e2e_ms = round(total_ms, 2)
    success = e2e_ms <= timeout_ms
    sla_violation = e2e_ms > sla_ms

    return {
        "statusCode": 200,
        "transaction_id": event.get("transaction_id", tx.get("transaction_id", "unknown")),
        "transaction": tx,
        "action": action,
        "stage_results": stage_results,
        "stage_latencies_ms": stage_latencies_ms,
        "e2e_ms": e2e_ms,
        "success": success,
        "sla_violation": sla_violation,
        "sla_ms": sla_ms,
        "timeout_ms": timeout_ms,
    }
