import numpy as np

from attacks import AttackConfig, MetricReplayAttack, RandomObservationAttack, TelemetryPoison
from src.config import STAGES
from src.linkage_lib import QCOLS
from src.ml_model.data_api_env import DataApiRoutingEnv


def make_env(tmp_path):
    tx = {
        "slot": [0, 1],
        "window_id": [0, 1],
        "type": ["CASH_IN", "PAYMENT"],
        "amount": [10.0, 20.0],
        "fraud_risk": [0.1, 0.2],
        "split": ["train", "train"],
    }
    import pandas as pd
    import json

    tx_df = pd.DataFrame(tx)
    tx_df.to_parquet(tmp_path / "linkage_tx_sample.parquet")
    rows = []
    for stage in STAGES:
        for candidate in range(3):
            for window in range(2):
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
    pd.DataFrame({"slot": [0, 1], "window_id": [0, 1]}).to_parquet(tmp_path / "slot_to_window.parquet")
    (tmp_path / "linkage_meta.json").write_text(json.dumps({"sla_ms": 100.0, "timeout_ms": 200.0}), encoding="utf-8")
    return tmp_path


def test_telemetry_poison_changes_observation(tmp_path):
    data_dir = make_env(tmp_path)
    env = DataApiRoutingEnv(data_dir=data_dir, split="train", episode_length=2, seed=7)
    cfg = AttackConfig(id="A1", seed=7, start=0, duration=5, rho=0.4, margin=0.1)
    attack = TelemetryPoison(env, cfg)
    obs, info = attack.reset(options={"start": 0})

    assert info["attack_active"] is True
    assert np.any(np.abs(obs - info["true_obs"]) > 0)


def test_random_attack_changes_observation(tmp_path):
    data_dir = make_env(tmp_path)
    env = DataApiRoutingEnv(data_dir=data_dir, split="train", episode_length=2, seed=7)
    cfg = AttackConfig(id="A2-random", seed=7, start=0, duration=5, epsilon=0.2)
    attack = RandomObservationAttack(env, cfg)
    obs, info = attack.reset(options={"start": 0})

    assert info["attack_active"] is True
    assert np.any(np.abs(obs - info["true_obs"]) > 0)


def test_metric_replay_uses_previous_obs(tmp_path):
    data_dir = make_env(tmp_path)
    env = DataApiRoutingEnv(data_dir=data_dir, split="train", episode_length=2, seed=7)
    cfg = AttackConfig(id="A6", seed=7, start=0, duration=5, delta=1)
    attack = MetricReplayAttack(env, cfg)
    obs1, _ = attack.reset(options={"start": 0})
    action = np.zeros(len(STAGES), dtype=np.int64)
    obs2, _, _, _, info = attack.step(action)

    assert info["attack_active"] is True
    assert obs2.shape == obs1.shape
