"""SageMaker inference entry point for the trained Role 1 PPO policy.

The request accepts either ``{"instances": [{"observation": [...]}]}`` or
``{"instances": [[...]]}``. An optional context object can contain
``e2e_ms``, ``sla_ms``, ``timeout_ms`` and ``fraud_risk`` so the response can
explain the realized reward after the transaction completes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

STAGES = ("ingress", "auth", "routing", "processor", "db")
METRICS_PER_CANDIDATE = 5
CANDIDATES_PER_STAGE = 3
DEFAULT_SLA_MS = 1_000.0
DEFAULT_TIMEOUT_MS = 2_000.0


def model_fn(model_dir: str) -> PPO:
    """Load the SB3 artifact from SageMaker's model directory."""
    root = Path(model_dir)
    candidates = (root / "data_ppo_agent.zip", root / "model.zip", root / "data_ppo_agent")
    model_path = next((path for path in candidates if path.exists()), None)
    if model_path is None:
        raise FileNotFoundError(
            f"No PPO artifact found in {root}; expected data_ppo_agent.zip or model.zip"
        )
    return PPO.load(str(model_path), device="cpu")


def _as_request_instances(payload: Any) -> list[tuple[np.ndarray, dict[str, Any]]]:
    if isinstance(payload, dict):
        instances = payload.get("instances")
        if instances is None:
            instances = [payload]
    elif isinstance(payload, list):
        instances = payload
    else:
        raise ValueError("Request must be a JSON object or array")

    parsed = []
    for instance in instances:
        if isinstance(instance, dict):
            observation = instance.get("observation")
            context = instance.get("context") or {}
        else:
            observation = instance
            context = {}
        if observation is None:
            raise ValueError("Each instance must contain an observation")
        values = np.asarray(observation, dtype=np.float32)
        if values.ndim != 1 or values.size != 83:
            raise ValueError("Each observation must contain exactly 83 numeric values")
        parsed.append((values, context))
    return parsed


def _reward_explanation(
    observation: np.ndarray,
    action: np.ndarray,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Return auditable reward terms and the largest absolute contributor."""
    sla_ms = float(context.get("sla_ms", os.getenv("SLA_MS", DEFAULT_SLA_MS)))
    timeout_ms = float(context.get("timeout_ms", os.getenv("TIMEOUT_MS", DEFAULT_TIMEOUT_MS)))
    fraud_risk = float(context.get("fraud_risk", observation[-2]))
    e2e_ms = context.get("e2e_ms")
    estimated = e2e_ms is None
    if estimated:
        e2e_ms = sum(
            float(observation[stage * 15 + int(candidate) * METRICS_PER_CANDIDATE + 1])
            * timeout_ms
            for stage, candidate in enumerate(action)
        )
    e2e_ms = float(e2e_ms)
    terms = {
        "success_bonus": 1.0 if e2e_ms <= timeout_ms else -1.0,
        "latency_penalty": -e2e_ms / sla_ms,
        "sla_violation_penalty": -0.75 if e2e_ms > sla_ms else 0.0,
        "fraud_risk_bonus": 0.20 * (1.0 - fraud_risk),
    }
    dominant = max(terms, key=lambda name: abs(terms[name]))
    return {
        "terms": terms,
        "dominant_reward_component": dominant,
        "dominant_value": terms[dominant],
        "e2e_ms": e2e_ms,
        "estimated": estimated,
    }


def predict_fn(input_data: Any, model: PPO) -> list[dict[str, Any]]:
    """Predict one routing action and explanation per request instance."""
    predictions = []
    for observation, context in _as_request_instances(input_data):
        action, _ = model.predict(observation, deterministic=True)
        action_array = np.asarray(action, dtype=np.int64).reshape(-1)
        if action_array.size != len(STAGES):
            raise ValueError("The loaded policy did not return one action per stage")
        predictions.append(
            {
                "action": {stage: int(action_array[index]) for index, stage in enumerate(STAGES)},
                "explanation": _reward_explanation(observation, action_array, context),
            }
        )
    return predictions


def input_fn(request_body: str, request_content_type: str) -> Any:
    if request_content_type != "application/json":
        raise ValueError("Only application/json requests are supported")
    return json.loads(request_body)


def output_fn(prediction: Any, response_content_type: str) -> tuple[str, str]:
    if response_content_type != "application/json":
        raise ValueError("Only application/json responses are supported")
    return json.dumps({"predictions": prediction}), response_content_type