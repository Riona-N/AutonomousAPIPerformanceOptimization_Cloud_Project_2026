from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency handling
    torch = None

from .base import AttackWrapper


class TelemetryPoison(AttackWrapper):
    """A1: make attacker-controlled telemetry look better than it really is."""

    def on_reset(self) -> None:
        stage_count = self.stage_count
        target_count = max(1, int(round(getattr(self.cfg, "rho", 0.2) * stage_count)))
        self.target_stages = np.sort(self.rng.choice(stage_count, size=target_count, replace=False))
        self.attacker_candidates.clear()
        self.best_candidates.clear()

        if hasattr(self.env, "latency_model") and hasattr(self.env.latency_model, "p95"):
            p95 = np.asarray(self.env.latency_model.p95, dtype=np.float32)
            for stage_idx in self.target_stages:
                stage_values = p95[stage_idx].mean(axis=1)
                self.best_candidates[int(stage_idx)] = int(np.argmin(stage_values))
                self.attacker_candidates[int(stage_idx)] = int(np.argmax(stage_values))
        else:
            for stage_idx in self.target_stages:
                self.best_candidates[int(stage_idx)] = 0
                self.attacker_candidates[int(stage_idx)] = min(self.candidate_count - 1, 2)

    def perturb(self, obs: np.ndarray) -> np.ndarray:
        out = obs.copy()
        for stage_idx in self.target_stages:
            attacker = self.attacker_candidates.get(int(stage_idx))
            best = self.best_candidates.get(int(stage_idx))
            if attacker is None or best is None:
                continue
            for metric_idx in range(3):
                idx = self._metric_idx(int(stage_idx), attacker, metric_idx)
                best_idx = self._metric_idx(int(stage_idx), best, metric_idx)
                if getattr(self.cfg, "mode", "copy-best") == "bias":
                    out[idx] = np.clip(out[idx] - float(getattr(self.cfg, "epsilon", 0.1)), 0.0, 1.0)
                else:
                    out[idx] = np.clip(out[best_idx] - float(getattr(self.cfg, "margin", 0.05)), 0.0, 1.0)
            out[self._metric_idx(int(stage_idx), attacker, 4)] = 0.0
        return out


class RandomObservationAttack(AttackWrapper):
    """A2 random perturbation: add bounded random noise to the observation."""

    def perturb(self, obs: np.ndarray) -> np.ndarray:
        delta = self.rng.uniform(
            -float(getattr(self.cfg, "epsilon", 0.1)),
            float(getattr(self.cfg, "epsilon", 0.1)),
            size=obs.shape,
        )
        return np.clip(obs + delta, 0.0, 1.0)


class FGSMObservationAttack(AttackWrapper):
    """A2 FGSM: use a policy gradient sign step to push the action away from its clean choice."""

    def perturb(self, obs: np.ndarray) -> np.ndarray:
        if getattr(self.cfg, "policy", None) is None or torch is None:
            return RandomObservationAttack.perturb(self, obs)

        policy = self.cfg.policy
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).clone()
        x = x.requires_grad_(True)
        action, _ = policy.predict(obs, deterministic=True)
        action_tensor = torch.as_tensor(np.asarray(action), dtype=torch.int64).reshape(1, -1)

        dist = policy.policy.get_distribution(x)
        loss = -dist.log_prob(action_tensor).sum()
        grad = torch.autograd.grad(loss, x)[0]
        delta = float(getattr(self.cfg, "epsilon", 0.1)) * grad.sign()
        return np.clip((x.detach() + delta).squeeze(0).numpy(), 0.0, 1.0)


class PGDObservationAttack(AttackWrapper):
    """A2 PGD: iterative projected gradient attack with a small step size."""

    def perturb(self, obs: np.ndarray) -> np.ndarray:
        if getattr(self.cfg, "policy", None) is None or torch is None:
            return RandomObservationAttack.perturb(self, obs)

        policy = self.cfg.policy
        x0 = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).clone()
        x = x0.clone().detach().requires_grad_(True)
        action, _ = policy.predict(obs, deterministic=True)
        action_tensor = torch.as_tensor(np.asarray(action), dtype=torch.int64).reshape(1, -1)
        eps = float(getattr(self.cfg, "epsilon", 0.1))
        steps = int(getattr(self.cfg, "steps", 10))
        step_size = float(getattr(self.cfg, "step_size", eps / 4.0))

        for _ in range(steps):
            x.requires_grad_(True)
            dist = policy.policy.get_distribution(x)
            loss = -dist.log_prob(action_tensor).sum()
            grad = torch.autograd.grad(loss, x)[0]
            x = x - step_size * grad.sign()
            x = torch.maximum(x0 - eps, torch.minimum(x, x0 + eps))
        return np.clip(x.detach().squeeze(0).numpy(), 0.0, 1.0)


class MetricReplayAttack(AttackWrapper):
    """A6: replay a healthy latency block from a delta window earlier."""

    def perturb(self, obs: np.ndarray) -> np.ndarray:
        if len(self.history) <= 1:
            return obs.copy()
        delta = int(getattr(self.cfg, "delta", 2))
        if len(self.history) <= delta:
            return obs.copy()
        source = self.history[-(delta + 1)]
        out = obs.copy()
        block_size = self._feature_block_size()
        if source.shape[0] < block_size:
            return out
        out[:block_size] = source[:block_size]
        return out
