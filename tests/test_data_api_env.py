import json

import numpy as np
import pandas as pd
import pytest

from src.config import CANDIDATES_PER_STAGE, STAGES
from src.ml_model.data_api_env import DataApiRoutingEnv
from src.linkage_lib import QCOLS


@pytest.fixture
def routing_data(tmp_path):
    transactions = pd.DataFrame(
        {
            "slot": [0, 1, 2],
            "window_id": [0, 1, 2],
            "type": ["CASH_IN"] * 3,
            "amount": [10.0] * 3,
            "fraud_risk": [0.1] * 3,
            "split": ["train"] * 3,
        }
    )
    transactions.to_parquet(tmp_path / "linkage_tx_sample.parquet")

    rows = []
    for stage in STAGES:
        for candidate in range(3):
            for window in range(3):
                row = {
                    "stage": stage,
                    "cand_rank": candidate,
                    "window_id": window,
                    "n_calls": 10,
                    "p50": 1.0,
                    "p95": 2.0,
                    "p99": 3.0,
                    "sla_violation": 0,
                }
                row.update({column: float(index + 1) for index, column in enumerate(QCOLS)})
                rows.append(row)
    pd.DataFrame(rows).to_parquet(tmp_path / "stage_latency_windows.parquet")
    pd.DataFrame(
        {"slot": [0, 1, 2], "window_id": [0, 1, 2]}
    ).to_parquet(tmp_path / "slot_to_window.parquet")
    (tmp_path / "linkage_meta.json").write_text(
        json.dumps({"sla_ms": 100.0, "timeout_ms": 200.0}),
        encoding="utf-8",
    )
    return tmp_path


def make_env(data_dir, episode_length=3):
    return DataApiRoutingEnv(
        data_dir=data_dir,
        split="train",
        episode_length=episode_length,
        seed=7,
    )


def test_slot_mapping_accepts_first_and_last_slots(routing_data):
    env = make_env(routing_data)

    assert env._window_for_slot(0) == 0
    assert env._window_for_slot(2) == 2

    env.reset(options={"start": 0})
    make_env(routing_data, episode_length=1).reset(options={"start": 2})


def test_reset_rejects_missing_slot_mapping(routing_data):
    env = make_env(routing_data)
    del env._slot_to_window[0]

    with pytest.raises(ValueError, match="No window mapping found for slot=0"):
        env.reset(options={"start": 0})


def test_step_rejects_missing_slot_mapping(routing_data):
    env = make_env(routing_data)
    env.reset(options={"start": 0})
    del env._slot_to_window[0]

    with pytest.raises(ValueError, match="No window mapping found for slot=0"):
        env.step(np.zeros(len(STAGES), dtype=np.int64))


def test_reset_rejects_window_not_in_latency_model(routing_data):
    env = make_env(routing_data)
    env._slot_to_window[0] = 99

    with pytest.raises(ValueError, match="Slot 0 maps to an invalid window"):
        env.reset(options={"start": 0})


def test_observation_shape_uses_configured_candidate_count(routing_data):
    env = make_env(routing_data)

    expected = (
        len(STAGES) * CANDIDATES_PER_STAGE * 5
        + len(env.transaction_types)
        + 3
    )
    assert env.candidates_per_stage == CANDIDATES_PER_STAGE
    assert env.observation_space.shape == (expected,)


def test_latency_multiplier_scale_is_applied(routing_data):
    env = DataApiRoutingEnv(
        data_dir=routing_data,
        split="train",
        episode_length=1,
        seed=7,
        latency_multiplier=0.0,
    )
    env.reset(options={"start": 0})

    _, _, _, _, info = env.step(np.zeros(len(STAGES), dtype=np.int64))
    assert np.allclose(info["stage_latencies_ms"], 0.0)
    assert info["e2e_ms"] == 0.0


def test_step_rejects_after_termination(routing_data):
    env = make_env(routing_data, episode_length=1)
    env.reset(options={"start": 0})

    _, _, terminated, _, _ = env.step(np.zeros(len(STAGES), dtype=np.int64))
    assert terminated is True

    with pytest.raises(RuntimeError, match=r"Cannot call step\(\) after termination"):
        env.step(np.zeros(len(STAGES), dtype=np.int64))


def test_d4_reward_weights_are_zero_by_default(routing_data):
    env = make_env(routing_data)
    env.reset(options={"start": 0})

    _, reward, _, _, info = env.step(np.zeros(len(STAGES), dtype=np.int64))

    assert reward == info["reward_terms"]["base_reward"]
    assert info["reward_terms"]["cost_penalty"] == 0.0
    assert info["reward_terms"]["low_trust_penalty"] == 0.0
    assert info["reward_terms"]["fraud_value_penalty"] == 0.0


def test_d4_reward_applies_low_trust_and_labeled_low_risk_penalties(routing_data):
    tx_path = routing_data / "linkage_tx_sample.parquet"
    transactions = pd.read_parquet(tx_path)
    transactions["is_fraud"] = [1, 0, 0]
    transactions.to_parquet(tx_path)
    env = DataApiRoutingEnv(
        data_dir=routing_data,
        split="train",
        episode_length=3,
        seed=7,
        reward_beta=0.8,
        reward_gamma=0.5,
        reward_delta=0.7,
    )
    env.reset(options={"start": 0})

    _, reward, _, _, info = env.step(np.zeros(len(STAGES), dtype=np.int64))
    terms = info["reward_terms"]

    assert terms["cost"] == 0.0
    assert terms["cost_penalty"] == 0.0
    assert terms["low_trust_fraction"] == 1.0
    assert terms["low_trust_penalty"] == 0.5
    assert terms["normalized_fraud_value"] == 1.0
    assert terms["fraud_value_penalty"] == 0.7
    assert reward == pytest.approx(terms["base_reward"] - 1.2)


def test_d4_reward_rejects_negative_weights_or_invalid_threshold(routing_data):
    with pytest.raises(ValueError, match="weights"):
        DataApiRoutingEnv(data_dir=routing_data, reward_gamma=-0.1)
    with pytest.raises(ValueError, match="threshold"):
        DataApiRoutingEnv(data_dir=routing_data, reward_trust_threshold=1.1)
