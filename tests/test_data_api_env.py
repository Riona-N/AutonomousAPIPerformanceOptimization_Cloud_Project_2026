import json

import numpy as np
import pandas as pd
import pytest

from src.config import STAGES
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


def test_step_rejects_after_termination(routing_data):
    env = make_env(routing_data, episode_length=1)
    env.reset(options={"start": 0})

    _, _, terminated, _, _ = env.step(np.zeros(len(STAGES), dtype=np.int64))
    assert terminated is True

    with pytest.raises(RuntimeError, match=r"Cannot call step\(\) after termination"):
        env.step(np.zeros(len(STAGES), dtype=np.int64))
