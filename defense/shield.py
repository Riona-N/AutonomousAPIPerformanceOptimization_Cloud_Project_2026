from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class ShieldResult:
    action: np.ndarray
    overridden_stages: tuple[int, ...]
    alerts: tuple[int, ...]
    fallback_stages: tuple[int, ...]


class SafetyShield:
    """Override untrusted routing choices with a latency-based safe choice."""

    def __init__(
        self,
        threshold: float = 0.5,
        hold_windows: int = 3,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be within [0, 1]")
        if hold_windows < 0:
            raise ValueError("hold_windows must be non-negative")
        self.threshold = float(threshold)
        self.hold_windows = int(hold_windows)
        self._hold_remaining: dict[int, int] = {}
        self._last_window_id: int | None = None

    def reset(self) -> None:
        self._hold_remaining.clear()
        self._last_window_id = None

    def apply(
        self,
        action: np.ndarray | list[int] | tuple[int, ...],
        trust_scores: np.ndarray,
        level_p95: np.ndarray,
        *,
        window_id: int | None = None,
    ) -> ShieldResult:
        proposed = np.asarray(action, dtype=np.int64)
        trust = np.asarray(trust_scores, dtype=np.float64)
        p95 = np.asarray(level_p95, dtype=np.float64)
        if proposed.ndim != 1:
            raise ValueError("action must be a one-dimensional candidate index per stage")
        if trust.ndim != 2 or p95.shape != trust.shape:
            raise ValueError("trust_scores and level_p95 must have matching (stages, candidates) shapes")
        if trust.shape[0] != proposed.size or trust.shape[1] == 0:
            raise ValueError("action stages must match non-empty trust candidate rows")
        if not np.isfinite(trust).all() or not np.isfinite(p95).all():
            raise ValueError("trust_scores and level_p95 must contain only finite values")
        if np.any((trust < 0.0) | (trust > 1.0)):
            raise ValueError("trust_scores must be within [0, 1]")
        if np.any((proposed < 0) | (proposed >= trust.shape[1])):
            raise ValueError("action contains a candidate index outside the trust matrix")
        if np.any(p95 < 0.0):
            raise ValueError("level_p95 values must be non-negative")

        self._advance_window(window_id)
        selected = proposed.copy()
        overridden: list[int] = []
        alerts: list[int] = []
        fallback: list[int] = []
        for stage, candidate in enumerate(proposed):
            is_holding = self._hold_remaining.get(stage, 0) > 0
            chosen_trust = trust[stage, int(candidate)]
            if is_holding:
                fallback.append(stage)
                selected[stage] = self._fallback_candidate(stage, trust, p95, alerts)
                overridden.append(stage)
                continue
            if chosen_trust < self.threshold:
                selected[stage] = self._fallback_candidate(stage, trust, p95, alerts)
                overridden.append(stage)
                fallback.append(stage)
                if self.hold_windows:
                    self._hold_remaining[stage] = self.hold_windows + 1

        return ShieldResult(
            action=selected,
            overridden_stages=tuple(overridden),
            alerts=tuple(alerts),
            fallback_stages=tuple(fallback),
        )

    def _advance_window(self, window_id: int | None) -> None:
        if window_id is None:
            for stage in tuple(self._hold_remaining):
                self._hold_remaining[stage] -= 1
                if self._hold_remaining[stage] <= 0:
                    del self._hold_remaining[stage]
            return

        current = int(window_id)
        if current < 0:
            raise ValueError("window_id must be non-negative")
        if self._last_window_id is not None and current < self._last_window_id:
            raise ValueError("window_id must be monotonically non-decreasing")
        if self._last_window_id is not None and current > self._last_window_id:
            elapsed = current - self._last_window_id
            for stage in tuple(self._hold_remaining):
                self._hold_remaining[stage] -= elapsed
                if self._hold_remaining[stage] <= 0:
                    del self._hold_remaining[stage]
        self._last_window_id = current

    def _fallback_candidate(
        self,
        stage: int,
        trust: np.ndarray,
        p95: np.ndarray,
        alerts: list[int],
    ) -> int:
        trusted = np.flatnonzero(trust[stage] >= self.threshold)
        if trusted.size == 0:
            alerts.append(stage)
            candidates = np.arange(trust.shape[1])
        else:
            candidates = trusted
        return int(candidates[np.argmin(p95[stage, candidates])])


TrustProvider = Callable[[np.ndarray], np.ndarray]


class SafetyShieldWrapper(gym.Wrapper):
    """Apply a safety shield to policy actions using trust in its latest observation.

    By default, trust scores are read from the 15 trust features appended to an
    83-feature observation. Other observations must provide a ``trust_provider``
    callback. The latency feature layout is stage/candidate blocks of five, with
    normalized p95 at metric index 1.
    """

    def __init__(
        self,
        env: gym.Env,
        shield: SafetyShield | None = None,
        trust_provider: TrustProvider | None = None,
        window_id_provider: Callable[[np.ndarray], int] | None = None,
    ) -> None:
        super().__init__(env)
        self.shield = shield or SafetyShield()
        self.trust_provider = trust_provider
        self.window_id_provider = window_id_provider
        self._observation: np.ndarray | None = None
        self._decision_window = -1
        action_shape = getattr(env.action_space, "nvec", None)
        self._stage_count = len(action_shape) if action_shape is not None else 0
        observation_shape = getattr(env.observation_space, "shape", None)
        self._observation_size = int(np.prod(observation_shape)) if observation_shape else 0
        self._candidate_count = int(action_shape[0]) if action_shape is not None and len(action_shape) else 0

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        self.shield.reset()
        self._decision_window = -1
        self._observation = np.asarray(observation, dtype=np.float32)
        return self._observation.copy(), info

    def step(self, action: np.ndarray | list[int] | tuple[int, ...]):
        if self._observation is None:
            raise RuntimeError("Call reset() before stepping the shield wrapper")
        trust, p95 = self._signals(self._observation)
        if self.window_id_provider is not None:
            window_id = int(self.window_id_provider(self._observation))
        else:
            self._decision_window += 1
            window_id = self._decision_window
        result = self.shield.apply(action, trust, p95, window_id=window_id)
        observation, reward, terminated, truncated, info = self.env.step(result.action)
        self._observation = np.asarray(observation, dtype=np.float32)
        info["shield"] = {
            "overridden_stages": list(result.overridden_stages),
            "fallback_stages": list(result.fallback_stages),
            "alerts": list(result.alerts),
            "action": result.action.copy(),
        }
        return self._observation.copy(), reward, terminated, truncated, info

    def _signals(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._stage_count == 0 or self._candidate_count == 0:
            raise TypeError("SafetyShieldWrapper requires a MultiDiscrete stage action space")
        if self.trust_provider is not None:
            trust = np.asarray(self.trust_provider(observation), dtype=np.float64)
        else:
            trust_count = self._stage_count * self._candidate_count
            if observation.size < trust_count:
                raise ValueError("Observation does not contain appended trust scores")
            trust = observation[-trust_count:].reshape(
                self._stage_count, self._candidate_count
            ).astype(np.float64)

        metric_count = 5
        metric_block_size = self._stage_count * self._candidate_count * metric_count
        if observation.size < metric_block_size:
            raise ValueError("Observation is too small to contain the latency metric block")
        metrics = observation[:metric_block_size].reshape(
            self._stage_count, self._candidate_count, metric_count
        )
        level_p95 = metrics[:, :, 1].astype(np.float64)
        return trust, level_p95
