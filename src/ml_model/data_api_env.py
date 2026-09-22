"""Data-backed Gymnasium environment for Role 1's offline routing baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

try:
    from src.config import OUT, STAGES
    from src.linkage_lib import LatencyModel, QCOLS
except ImportError:
    from config import OUT, STAGES
    from linkage_lib import LatencyModel, QCOLS


class DataApiRoutingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Route each transaction through one candidate at each payment stage.

    Observation layout:
      5 stages x 3 candidates x (p50, p95, p99, log1p(n_calls), sla_flag),
      followed by 5 transaction-type flags, log amount, fraud risk, merchant flag.

    Metrics are deliberately one minute stale: the observation uses the previous
    slot's window while the sampled latency uses the transaction's current window.
    """

    metadata = {"render_modes": []}
    stages = tuple(STAGES)
    transaction_types = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
    candidates_per_stage = 3
    metrics_per_candidate = 5

    def __init__(
        self,
        data_dir: str | Path = OUT,
        split: str = "train",
        episode_length: int = 512,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.episode_length = episode_length
        self._seed = seed

        tx_path = self.data_dir / "linkage_tx.parquet"
        if not tx_path.exists():
            tx_path = self.data_dir / "linkage_tx_sample.parquet"
        tx = pd.read_parquet(tx_path)
        required = {"slot", "window_id", "type", "amount", "fraud_risk", "split"}
        missing = required.difference(tx.columns)
        if missing:
            raise ValueError(f"Transaction table is missing columns: {sorted(missing)}")
        self.transactions = tx[tx["split"] == split].sort_values("slot").reset_index(drop=True)
        if self.transactions.empty:
            raise ValueError(f"No transactions available for split={split!r}")

        metadata = json.loads((self.data_dir / "linkage_meta.json").read_text(encoding="utf-8"))
        self.sla_ms = float(metadata["sla_ms"])
        self.timeout_ms = float(metadata["timeout_ms"])
        self.latency_model = LatencyModel(self.data_dir / "stage_latency_windows.parquet")
        metrics = pd.read_parquet(self.data_dir / "stage_latency_windows.parquet")
        self._metric_values = self._build_metric_tensor(metrics)
        self._slot_to_window = pd.read_parquet(self.data_dir / "slot_to_window.parquet")[
            "window_id"
        ].to_numpy()
        self.amount_scale = max(
            1.0,
            float(np.nanpercentile(self.transactions["amount"].to_numpy(dtype=float), 99.9)),
        )

        observation_size = (
            len(self.stages) * self.candidates_per_stage * self.metrics_per_candidate
            + len(self.transaction_types)
            + 3
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(observation_size,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(
            np.full(len(self.stages), self.candidates_per_stage, dtype=np.int64)
        )
        self._position = 0
        self._episode_start = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed if seed is not None else self._seed)
        max_start = max(0, len(self.transactions) - self.episode_length)
        if options and "start" in options:
            start = int(options["start"])
            if not 0 <= start <= max_start:
                raise ValueError(f"start must be between 0 and {max_start}")
        else:
            start = int(self.np_random.integers(0, max_start + 1))
        self._episode_start = start
        self._position = 0
        return self._observation(), {"split": self.split, "position": self._position}

    def step(
        self, action: np.ndarray | list[int] | tuple[int, ...]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=np.int64)
        if not self.action_space.contains(action_array):
            raise ValueError("Action must contain one candidate index (0..2) per stage")

        row = self.transactions.iloc[self._episode_start + self._position]
        window_ids = np.full(len(self.stages), int(row["window_id"]), dtype=np.int64)
        uniforms = self.np_random.random(len(self.stages))
        stage_latencies = np.asarray(
            [
                self.latency_model.sample(
                    stage_idx=stage_idx,
                    cand=int(action_array[stage_idx]),
                    window_ids=window_ids[stage_idx : stage_idx + 1],
                    u=uniforms[stage_idx : stage_idx + 1],
                )[0]
                for stage_idx in range(len(self.stages))
            ],
            dtype=np.float32,
        )
        e2e_ms = float(stage_latencies.sum())
        success = e2e_ms <= self.timeout_ms
        sla_violation = e2e_ms > self.sla_ms
        fraud_risk = float(row["fraud_risk"])
        reward = (
            (1.0 if success else -1.0)
            - e2e_ms / self.sla_ms
            - 0.75 * float(sla_violation)
            + 0.20 * (1.0 - fraud_risk)
        )

        self._position += 1
        terminated = self._position >= min(
            self.episode_length, len(self.transactions) - self._episode_start
        )
        info = {
            "e2e_ms": e2e_ms,
            "stage_latencies_ms": stage_latencies,
            "success": success,
            "sla_violation": sla_violation,
            "fraud_risk": fraud_risk,
            "slot": int(row["slot"]),
            "window_id": int(row["window_id"]),
            "action": action_array.copy(),
        }
        return self._observation(), float(reward), terminated, False, info

    def _build_metric_tensor(self, metrics: pd.DataFrame) -> np.ndarray:
        windows = self.latency_model.windows
        tensor = np.zeros(
            (len(self.stages), self.candidates_per_stage, len(windows), self.metrics_per_candidate),
            dtype=np.float32,
        )
        stage_index = {stage: index for index, stage in enumerate(self.stages)}
        window_index = {int(window): index for index, window in enumerate(windows)}
        for row in metrics.itertuples(index=False):
            if row.stage not in stage_index or int(row.cand_rank) >= self.candidates_per_stage:
                continue
            stage = stage_index[row.stage]
            candidate = int(row.cand_rank)
            window = window_index[int(row.window_id)]
            tensor[stage, candidate, window] = (
                np.clip(float(row.p50) / self.timeout_ms, 0.0, 1.0),
                np.clip(float(row.p95) / self.timeout_ms, 0.0, 1.0),
                np.clip(float(row.p99) / self.timeout_ms, 0.0, 1.0),
                np.clip(np.log1p(float(row.n_calls)) / 10.0, 0.0, 1.0),
                float(row.sla_violation),
            )
        return tensor

    def _observation(self) -> np.ndarray:
        row_index = min(
            self._episode_start + self._position,
            len(self.transactions) - 1,
        )
        row = self.transactions.iloc[row_index]
        slot = int(row["slot"])
        previous_slot = max(0, slot - 1)
        previous_window = int(self._slot_to_window[previous_slot])
        window_index = int(np.searchsorted(self.latency_model.windows, previous_window))
        metric_features = self._metric_values[:, :, window_index, :].reshape(-1)
        type_features = np.asarray(
            [float(row["type"] == transaction_type) for transaction_type in self.transaction_types],
            dtype=np.float32,
        )
        transaction_features = np.asarray(
            [
                np.clip(np.log1p(float(row["amount"])) / np.log1p(self.amount_scale), 0.0, 1.0),
                np.clip(float(row["fraud_risk"]), 0.0, 1.0),
                float(str(row["type"]).startswith("TRANSFER")),
            ],
            dtype=np.float32,
        )
        return np.concatenate([metric_features, type_features, transaction_features]).astype(
            np.float32
        )