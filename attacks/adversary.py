from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def apply_bounded_perturbation(
    observation: np.ndarray,
    perturbation: np.ndarray,
    epsilon: float,
    observation_space: spaces.Box,
) -> np.ndarray:
    """Apply an L-infinity bounded perturbation while preserving observation bounds."""
    clean = np.asarray(observation, dtype=np.float32)
    delta = np.clip(np.asarray(perturbation, dtype=np.float32), -epsilon, epsilon)
    exceeds_delta_bound = np.abs(delta.astype(np.float64)) > epsilon
    if np.any(exceeds_delta_bound):
        delta[exceeds_delta_bound] = np.nextafter(delta[exceeds_delta_bound], 0.0)
    attacked = np.clip(
        clean.astype(np.float64) + delta.astype(np.float64),
        observation_space.low,
        observation_space.high,
    ).astype(np.float32)

    exceeds_bound = np.abs(attacked.astype(np.float64) - clean.astype(np.float64)) > epsilon
    if np.any(exceeds_bound):
        closer = np.nextafter(attacked[exceeds_bound], clean[exceeds_bound])
        still_exceeds = (
            np.abs(closer.astype(np.float64) - clean[exceeds_bound].astype(np.float64))
            > epsilon
        )
        closer[still_exceeds] = clean[exceeds_bound][still_exceeds]
        attacked[exceeds_bound] = closer
    return attacked


class LearnedAdversaryEnv(gym.Env[np.ndarray, np.ndarray]):
    """Train an observation attacker against a fixed routing policy."""

    metadata = {"render_modes": []}

    def __init__(self, env: gym.Env, frozen_router, epsilon: float = 0.1) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be greater than zero")
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("The routing environment must have a Box observation space")

        self.env = env
        self.frozen_router = frozen_router
        router_policy = getattr(frozen_router, "policy", None)
        if router_policy is not None:
            router_policy.set_training_mode(False)
            for parameter in router_policy.parameters():
                parameter.requires_grad_(False)
        self.epsilon = float(epsilon)
        self.observation_space = env.observation_space
        self.action_space = spaces.Box(
            low=-self.epsilon,
            high=self.epsilon,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )
        self._observation: np.ndarray | None = None

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        observation, info = self.env.reset(seed=seed, options=options)
        self._observation = np.asarray(observation, dtype=np.float32)
        return self._observation.copy(), info

    def step(self, perturbation: np.ndarray):
        delta = np.asarray(perturbation, dtype=np.float32)
        if not self.action_space.contains(delta):
            raise ValueError("Adversary action must be an epsilon-bounded perturbation vector")

        if self._observation is None:
            raise RuntimeError("Call reset() before step()")
        clean_observation = self._observation
        attacked_observation = self._apply_perturbation(clean_observation, delta)
        router_action, _ = self.frozen_router.predict(
            attacked_observation, deterministic=True
        )
        next_observation, router_reward, terminated, truncated, info = self.env.step(
            router_action
        )
        self._observation = np.asarray(next_observation, dtype=np.float32)
        info["router_reward"] = float(router_reward)
        info["adversary_reward"] = -float(router_reward)
        info["perturbation_linf"] = float(
            np.max(np.abs(attacked_observation - clean_observation), initial=0.0)
        )
        return (
            self._observation.copy(),
            -float(router_reward),
            terminated,
            truncated,
            info,
        )

    def _apply_perturbation(
        self, observation: np.ndarray, perturbation: np.ndarray
    ) -> np.ndarray:
        return apply_bounded_perturbation(
            observation, perturbation, self.epsilon, self.observation_space
        )

    def close(self) -> None:
        self.env.close()
