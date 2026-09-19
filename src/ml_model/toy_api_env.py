"""A small Gymnasium environment for learning API backend routing."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ToyApiRoutingEnv(gym.Env[np.ndarray, int]):
    """Route requests to one of three simulated backends.

    The observation contains each backend's normalized latency, current load,
    and success rate, followed by the current request's normalized risk signal.
    The agent receives a larger reward for fast, successful, non-overloaded
    requests. This is intentionally small and deterministic enough to inspect
    while still requiring a policy to adapt to changing backend conditions.
    """

    metadata = {"render_modes": []}

    def __init__(self, episode_length: int = 100, seed: int | None = None) -> None:
        super().__init__()
        self.backend_count = 3
        self.episode_length = episode_length
        self.action_space = spaces.Discrete(self.backend_count)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.backend_count * 3 + 1,), dtype=np.float32
        )
        self._seed = seed
        self._step = 0
        self._latency = np.zeros(self.backend_count, dtype=np.float32)
        self._load = np.zeros(self.backend_count, dtype=np.float32)
        self._success_rate = np.zeros(self.backend_count, dtype=np.float32)
        self._request_risk = 0.0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed if seed is not None else self._seed)
        self._step = 0
        self._latency = self.np_random.uniform(0.15, 0.75, self.backend_count).astype(
            np.float32
        )
        self._load = self.np_random.uniform(0.1, 0.65, self.backend_count).astype(
            np.float32
        )
        self._success_rate = np.clip(
            1.0 - self._latency * 0.35 - self._load * 0.2, 0.55, 0.99
        ).astype(np.float32)
        self._request_risk = float(self.np_random.uniform(0.0, 1.0))
        return self._observation(), {"step": self._step}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be an integer in [0, {self.backend_count - 1}]")

        backend = int(action)
        overload = max(0.0, float(self._load[backend]) - 0.8)
        success_probability = np.clip(
            self._success_rate[backend] - overload * 0.7 - self._request_risk * 0.08,
            0.05,
            0.99,
        )
        success = bool(self.np_random.random() < success_probability)
        latency = float(self._latency[backend] + self._load[backend] * 0.35)
        reward = (1.0 if success else -1.0) - latency - overload * 2.0
        reward += 0.25 * float(self._success_rate[backend])

        self._step += 1
        self._load = np.clip(
            self._load + self.np_random.normal(0.0, 0.04, self.backend_count), 0.0, 1.0
        ).astype(np.float32)
        self._latency = np.clip(
            self._latency + self.np_random.normal(0.0, 0.03, self.backend_count), 0.05, 1.0
        ).astype(np.float32)
        self._success_rate = np.clip(
            1.0 - self._latency * 0.35 - self._load * 0.2, 0.05, 0.99
        ).astype(np.float32)
        self._request_risk = float(self.np_random.uniform(0.0, 1.0))

        terminated = self._step >= self.episode_length
        info = {
            "backend": backend,
            "success": success,
            "latency": latency,
            "sla_violation": latency > 0.8,
        }
        return self._observation(), float(reward), terminated, False, info

    def _observation(self) -> np.ndarray:
        return np.concatenate(
            [self._latency, self._load, self._success_rate, np.array([self._request_risk])]
        ).astype(np.float32)
