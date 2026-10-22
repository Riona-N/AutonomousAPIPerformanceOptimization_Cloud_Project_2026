from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass
class AttackConfig:
    id: str = "attack"
    seed: int = 0
    start: int = 0
    duration: int = 10
    epsilon: float = 0.1
    rho: float = 0.2
    margin: float = 0.05
    delta: int = 2
    alpha: float = 1.0
    mode: str = "copy-best"
    steps: int = 10
    step_size: float | None = None
    policy: Any | None = None
    risk_bias: float = 0.25


class AttackWrapper(gym.Wrapper):
    """Gym wrapper with attack timing, tracking, and observation perturbation hooks."""

    def __init__(self, env: gym.Env, cfg: AttackConfig | dict[str, Any]):
        super().__init__(env)
        self.cfg = self._coerce_cfg(cfg)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.t = 0
        self.history: list[np.ndarray] = []
        self.target_stages: np.ndarray | None = None
        self.attacker_candidates: dict[int, int] = {}
        self.best_candidates: dict[int, int] = {}

    @staticmethod
    def _coerce_cfg(cfg: AttackConfig | dict[str, Any]) -> AttackConfig:
        if isinstance(cfg, AttackConfig):
            return cfg
        if isinstance(cfg, dict):
            return AttackConfig(**cfg)
        raise TypeError(f"Unsupported attack configuration type: {type(cfg)!r}")

    @property
    def stage_count(self) -> int:
        return len(getattr(self.env, "stages", ())) or 1

    @property
    def candidate_count(self) -> int:
        return int(getattr(self.env, "candidates_per_stage", 3))

    @property
    def metric_count(self) -> int:
        return int(getattr(self.env, "metrics_per_candidate", 5))

    def active(self) -> bool:
        start = int(getattr(self.cfg, "start", 0))
        duration = int(getattr(self.cfg, "duration", 0))
        return start <= self.t < start + duration

    def _metric_idx(self, stage_idx: int, candidate_idx: int, metric_idx: int) -> int:
        return (
            stage_idx * self.candidate_count * self.metric_count
            + candidate_idx * self.metric_count
            + metric_idx
        )

    def _feature_block_size(self) -> int:
        return self.stage_count * self.candidate_count * self.metric_count

    def hit_attacker(self, action: np.ndarray | list[int] | tuple[int, ...]) -> bool:
        action_array = np.asarray(action, dtype=np.int64)
        if action_array.ndim == 0:
            return False
        stage_limit = min(len(action_array), self.stage_count)
        for stage_idx in range(stage_limit):
            attacker = self.attacker_candidates.get(stage_idx)
            if attacker is not None and int(action_array[stage_idx]) == attacker:
                return True
        return False

    def maybe_attack(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        info["true_obs"] = np.asarray(obs, dtype=np.float32).copy()
        seen = self.perturb(np.asarray(obs, dtype=np.float32).copy()) if self.active() else np.asarray(obs, dtype=np.float32).copy()
        info["seen_obs"] = seen
        return seen

    def before_step(self, action: np.ndarray | list[int] | tuple[int, ...]) -> None:
        return None

    def on_reset(self) -> None:
        return None

    def perturb(self, obs: np.ndarray) -> np.ndarray:
        return obs

    def reset(self, **kwargs):
        self.t = 0
        self.history.clear()
        observation, info = self.env.reset(**kwargs)
        self.on_reset()
        info["attack_id"] = getattr(self.cfg, "id", "attack")
        info["attack_active"] = self.active()
        observation = np.asarray(observation, dtype=np.float32)
        self.history.append(observation.copy())
        return self.maybe_attack(observation, info), info

    def step(self, action):
        self.before_step(action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.t += 1
        info["attack_id"] = getattr(self.cfg, "id", "attack")
        info["attack_active"] = self.active()
        info["routed_to_attacker"] = self.hit_attacker(action)
        self.history.append(np.asarray(observation, dtype=np.float32).copy())
        if len(self.history) > 64:
            self.history.pop(0)
        return self.maybe_attack(observation, info), reward, terminated, truncated, info
