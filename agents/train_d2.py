from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from attacks import LearnedAdversaryEnv
from src.ml_model.data_api_env import DataApiRoutingEnv


DEFAULT_VERSIONS = [
    "PPO-clean",
    "PPO-noise",
    "PPO-PGD",
    "PPO-ATLA",
    "PPO-trust",
    "PPO-trust-ATLA",
    "CQL",
    "IQL",
]


def _trust_scores(env: DataApiRoutingEnv, obs: np.ndarray) -> np.ndarray:
    metric_len = env.candidates_per_stage * env.metrics_per_candidate * len(env.stages)
    metric_block = obs[:metric_len]
    if metric_block.size == 0:
        return np.zeros(15, dtype=np.float32)
    metric_matrix = metric_block.reshape(len(env.stages), env.candidates_per_stage, env.metrics_per_candidate)
    p95_block = metric_matrix[:, :, 1]
    call_support = metric_matrix[:, :, 3]
    sla_compliance = 1.0 - np.clip(metric_matrix[:, :, 4], 0.0, 1.0)
    trust = np.clip(call_support * sla_compliance, 0.0, 1.0)
    return trust.reshape(-1).astype(np.float32)[:15]


class ObservationNoiseWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, epsilon: float = 0.1, seed: int = 0):
        super().__init__(env)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)

    def _perturb(self, observation: np.ndarray) -> np.ndarray:
        noise = self.rng.uniform(-self.epsilon, self.epsilon, size=observation.shape)
        return np.clip(np.asarray(observation, dtype=np.float32) + noise, 0.0, 1.0).astype(np.float32)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._perturb(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._perturb(observation), reward, terminated, truncated, info


class HalfStepNoiseWrapper(ObservationNoiseWrapper):
    def _perturb(self, observation: np.ndarray) -> np.ndarray:
        if self.rng.random() < 0.5:
            return super()._perturb(observation)
        return np.asarray(observation, dtype=np.float32)


class SelfPGDWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, epsilon: float = 0.1, probability: float = 0.5, seed: int = 0):
        super().__init__(env)
        self.epsilon = float(epsilon)
        self.probability = float(probability)
        self.rng = np.random.default_rng(seed)
        self.policy = None

    def _perturb(self, observation: np.ndarray) -> np.ndarray:
        clean = np.asarray(observation, dtype=np.float32)
        if self.policy is None or self.rng.random() >= self.probability:
            return clean

        x0 = torch.as_tensor(clean, dtype=torch.float32).unsqueeze(0)
        x = x0.clone().detach()
        action, _ = self.policy.predict(clean, deterministic=True)
        action_tensor = torch.as_tensor(np.asarray(action), dtype=torch.int64).reshape(1, -1)
        step_size = self.epsilon / 4.0
        for _ in range(10):
            x = x.detach().requires_grad_(True)
            distribution = self.policy.policy.get_distribution(x)
            log_probability = distribution.log_prob(action_tensor).sum()
            gradient = torch.autograd.grad(log_probability, x)[0]
            x = x - step_size * gradient.sign()
            x = torch.maximum(x0 - self.epsilon, torch.minimum(x, x0 + self.epsilon))
        return torch.clamp(x.detach(), 0.0, 1.0).squeeze(0).numpy()

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._perturb(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._perturb(observation), reward, terminated, truncated, info


class TrustObservationWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def _augment(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if hasattr(self.env, "latency_model") and hasattr(self.env, "transactions"):
            trust = _trust_scores(self.env, obs)
            return np.concatenate([obs, trust], axis=0).astype(np.float32)
        return obs.astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, info

    @property
    def observation_space(self):
        base = self.env.observation_space
        if hasattr(base, "shape"):
            return gym.spaces.Box(low=0.0, high=1.0, shape=(int(base.shape[0]) + 15,), dtype=np.float32)
        return base


class FrozenAdversaryWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, adversary_policy, epsilon: float = 0.1):
        super().__init__(env)
        self.adversary_policy = adversary_policy
        self.epsilon = float(epsilon)

    def _attack(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        perturbation, _ = self.adversary_policy.predict(obs, deterministic=True)
        delta = np.asarray(perturbation, dtype=np.float32).reshape(obs.shape)
        delta = np.clip(delta, -self.epsilon, self.epsilon)
        attacked = np.clip(obs + delta, 0.0, 1.0).astype(np.float32)
        return attacked

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._attack(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._attack(observation), reward, terminated, truncated, info


class CurveCallback(BaseCallback):
    def __init__(self, eval_env, save_dir: Path, log_every: int = 1000, eval_episodes: int = 5):
        super().__init__("MlpPolicy")
        self.eval_env = eval_env
        self.save_dir = Path(save_dir)
        self.log_every = int(log_every)
        self.eval_episodes = int(eval_episodes)
        self.history: list[dict[str, float]] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.log_every != 0:
            return True
        self._record_evaluation()
        return True

    def _on_training_end(self) -> None:
        self._record_evaluation(phase="post_training")

    def _record_evaluation(self, phase: str = "training") -> None:
        rewards = []
        successes = []
        for _ in range(self.eval_episodes):
            seed = int(self.model.env.envs[0].np_random.integers(0, 10_000))
            obs, _ = self.eval_env.reset(seed=seed)
            episode_reward = 0.0
            episode_successes = []
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += float(reward)
                episode_successes.append(float(info.get("success", 0.0)))
                if terminated or truncated:
                    break
            rewards.append(float(episode_reward))
            successes.extend(episode_successes)
        self.history.append({
            "step": int(self.num_timesteps),
            "mean_reward": float(np.mean(rewards)),
            "success_rate": float(np.mean(successes)),
            "phase": phase,
        })
        self.save_dir.mkdir(parents=True, exist_ok=True)
        with (self.save_dir / "curve.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.history[-1]) + "\n")
        checkpoint_dir = self.save_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        if phase == "training":
            self.model.save(str(checkpoint_dir / f"model_{self.num_timesteps}_steps"))


def _evaluate_saved_ppo_models(results_root: Path, versions: list[str], seeds: list[int]) -> None:
    for version in versions:
        if not version.startswith("PPO-"):
            continue
        for seed in seeds:
            version_dir = results_root / version / f"seed_{seed}"
            model_path = version_dir / "model.zip"
            if not model_path.exists():
                raise FileNotFoundError(f"Missing trained PPO checkpoint: {model_path}")
            model = PPO.load(str(model_path), device="auto")
            eval_env = _make_env(seed + 1000, version, split="val")
            if isinstance(eval_env, SelfPGDWrapper):
                eval_env.policy = model

            rewards = []
            successes = []
            for episode in range(3):
                observation, _ = eval_env.reset(seed=seed + 20_000 + episode)
                episode_reward = 0.0
                while True:
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = eval_env.step(action)
                    episode_reward += float(reward)
                    successes.append(float(info.get("success", 0.0)))
                    if terminated or truncated:
                        break
                rewards.append(episode_reward)
            eval_env.close()
            evaluation = {
                "step": int(model.num_timesteps),
                "mean_reward": float(np.mean(rewards)),
                "success_rate": float(np.mean(successes)),
                "phase": "post_training_final",
            }
            curve_path = version_dir / "curve.jsonl"
            existing = [
                json.loads(line)
                for line in curve_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            existing = [row for row in existing if row.get("phase") != "post_training_final"]
            existing.append(evaluation)
            curve_path.write_text(
                "".join(json.dumps(row) + "\n" for row in existing),
                encoding="utf-8",
            )
            metrics_path = version_dir / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["post_training_final_evaluation"] = evaluation
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            del model

        _plot_version(version, results_root, seeds)


def _make_env(seed: int, version: str, split: str = "train") -> gym.Env:
    env = DataApiRoutingEnv(split=split, seed=seed)
    if version == "PPO-noise":
        return ObservationNoiseWrapper(env, epsilon=0.1, seed=seed)
    if version == "PPO-PGD":
        return SelfPGDWrapper(env, epsilon=0.1, probability=0.5, seed=seed)
    if version == "PPO-trust":
        return TrustObservationWrapper(env)
    if version == "PPO-trust-ATLA":
        return TrustObservationWrapper(env)
    return env


def _train_clean_router(seed: int, timesteps: int, root: Path, trust: bool = False) -> PPO:
    version = "PPO-trust" if trust else "PPO-clean"
    env = _make_env(seed, version, split="train")
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        verbose=0,
        device="auto",
    )
    path = root / "router_seed" / version / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=True)
    eval_env = _make_env(seed + 1000, version, split="val")
    cb = CurveCallback(eval_env, path, log_every=max(500, timesteps // 4), eval_episodes=3)
    model.learn(total_timesteps=timesteps, callback=cb)
    model.save(str(path / "router_model"))
    env.close()
    eval_env.close()
    return model


def _train_version(version: str, seed: int, timesteps: int, results_root: Path) -> dict[str, float]:
    version_dir = results_root / version / f"seed_{seed}"
    version_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env(seed=seed, version=version)

    if version == "PPO-ATLA":
        router_steps = max(2048, timesteps // 2)
        adversary_steps = max(2048, timesteps // 2)
        router = _train_clean_router(seed, router_steps, results_root)
        adversary_env = LearnedAdversaryEnv(DataApiRoutingEnv(split="train", seed=seed), router, epsilon=0.1)
        adversary = PPO(
            "MlpPolicy",
            adversary_env,
            seed=seed,
            n_steps=2048,
            batch_size=64,
            learning_rate=3e-4,
            gamma=0.99,
            ent_coef=0.01,
            verbose=0,
            device="auto",
        )
        adversary.learn(total_timesteps=adversary_steps)
        env = FrozenAdversaryWrapper(DataApiRoutingEnv(split="train", seed=seed), adversary, epsilon=0.1)

    if version == "PPO-trust-ATLA":
        router_steps = max(2048, timesteps // 2)
        adversary_steps = max(2048, timesteps // 2)
        router = _train_clean_router(seed, router_steps, results_root, trust=True)
        adversary_env = LearnedAdversaryEnv(_make_env(seed, "PPO-trust"), router, epsilon=0.1)
        adversary = PPO(
            "MlpPolicy",
            adversary_env,
            seed=seed,
            n_steps=2048,
            batch_size=64,
            learning_rate=3e-4,
            gamma=0.99,
            ent_coef=0.01,
            verbose=0,
            device="auto",
        )
        adversary.learn(total_timesteps=adversary_steps)
        env = FrozenAdversaryWrapper(_make_env(seed, "PPO-trust"), adversary, epsilon=0.1)

    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        verbose=0,
        device="auto",
    )
    if isinstance(env, SelfPGDWrapper):
        env.policy = model
    eval_env = _make_env(seed + 1000, version, split="val")
    cb = CurveCallback(eval_env, version_dir, log_every=max(500, timesteps // 4), eval_episodes=3)
    model.learn(total_timesteps=timesteps, callback=cb)
    model.save(str(version_dir / "model"))
    env.close()
    eval_env.close()

    metrics = {
        "version": version,
        "seed": seed,
        "timesteps": int(timesteps),
        "observation_shape": list(model.policy.observation_space.shape),
    }
    (version_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _action_count(env: DataApiRoutingEnv) -> int:
    return int(env.candidates_per_stage ** len(env.stages))


def _encode_action(action: np.ndarray, env: DataApiRoutingEnv) -> int:
    return int(np.ravel_multi_index(action, (env.candidates_per_stage,) * len(env.stages)))


def _decode_action(action_id: int, env: DataApiRoutingEnv) -> np.ndarray:
    return np.asarray(
        np.unravel_index(action_id, (env.candidates_per_stage,) * len(env.stages)),
        dtype=np.int64,
    )


class OfflineQNetwork(nn.Module):
    def __init__(self, observation_size: int, action_count: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_count),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


def _collect_offline_data(seed: int, sample_count: int):
    env = DataApiRoutingEnv(split="train", seed=seed)
    rng = np.random.default_rng(seed)
    observation, _ = env.reset(seed=seed)
    states, actions, rewards, next_states, dones = [], [], [], [], []
    while len(states) < sample_count:
        if rng.random() < 0.5:
            action = np.zeros(len(env.stages), dtype=np.int64)
        else:
            action = env.action_space.sample()
        next_observation, reward, terminated, truncated, _ = env.step(action)
        states.append(observation.copy())
        actions.append(_encode_action(action, env))
        rewards.append(float(reward))
        next_states.append(next_observation.copy())
        dones.append(float(terminated or truncated))
        observation = next_observation
        if terminated or truncated:
            observation, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    env.close()
    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
        np.asarray(rewards, dtype=np.float32),
        np.asarray(next_states, dtype=np.float32),
        np.asarray(dones, dtype=np.float32),
    )


def _evaluate_offline_policy(
    env: DataApiRoutingEnv,
    q_network: OfflineQNetwork,
    actor: nn.Module | None,
    episodes: int,
    seed: int,
) -> float:
    episode_rewards = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        while True:
            with torch.no_grad():
                state = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
                values = actor(state) if actor is not None else q_network(state)
                action_id = int(values.argmax(dim=1).item())
            observation, reward, terminated, truncated, _ = env.step(
                _decode_action(action_id, env)
            )
            total_reward += float(reward)
            if terminated or truncated:
                break
        episode_rewards.append(total_reward)
    return float(np.mean(episode_rewards))


def _train_offline(version: str, seed: int, timesteps: int, results_root: Path) -> dict:
    version_dir = results_root / version / f"seed_{seed}"
    checkpoint_dir = version_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    states, actions, rewards, next_states, dones = _collect_offline_data(
        seed, max(1024, timesteps)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_env = DataApiRoutingEnv(split="train", seed=seed)
    action_count = _action_count(action_env)
    action_env.close()
    q_network = OfflineQNetwork(states.shape[1], action_count).to(device)
    target_network = OfflineQNetwork(states.shape[1], action_count).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = torch.optim.Adam(q_network.parameters(), lr=3e-4)
    actor = None
    value_network = None
    actor_optimizer = None
    value_optimizer = None
    if version == "IQL":
        actor = nn.Sequential(
            nn.Linear(states.shape[1], 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_count),
        ).to(device)
        value_network = nn.Sequential(
            nn.Linear(states.shape[1], 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        ).to(device)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
        value_optimizer = torch.optim.Adam(value_network.parameters(), lr=3e-4)

    history = []
    rng = np.random.default_rng(seed)
    log_every = max(100, timesteps // 10)
    for update in range(1, timesteps + 1):
        indices = rng.integers(0, len(states), size=min(64, len(states)))
        state_batch = torch.as_tensor(states[indices], device=device)
        action_batch = torch.as_tensor(actions[indices], device=device)
        reward_batch = torch.as_tensor(rewards[indices], device=device)
        next_batch = torch.as_tensor(next_states[indices], device=device)
        done_batch = torch.as_tensor(dones[indices], device=device)

        q_values = q_network(state_batch)
        selected_q = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if version == "IQL":
                target_values = value_network(next_batch).squeeze(1)
            else:
                target_values = target_network(next_batch).max(dim=1).values
            bellman_target = reward_batch + 0.99 * (1.0 - done_batch) * target_values
        q_loss = F.smooth_l1_loss(selected_q, bellman_target)

        if version == "CQL":
            conservative_penalty = (
                torch.logsumexp(q_values, dim=1) - selected_q
            ).mean()
            loss = q_loss + 0.1 * conservative_penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
        else:
            q_optimizer_loss = q_loss
            optimizer.zero_grad()
            q_optimizer_loss.backward()
            optimizer.step()
            with torch.no_grad():
                data_q = selected_q.detach()
            state_value = value_network(state_batch).squeeze(1)
            advantage = data_q - state_value
            expectile = 0.7
            weights = torch.where(
                advantage > 0, expectile, 1.0 - expectile
            )
            value_loss = (weights * advantage.square()).mean()
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()
            actor_logits = actor(state_batch)
            actor_weights = torch.exp((advantage / 3.0).clamp(max=20)).detach()
            actor_loss = (
                F.cross_entropy(actor_logits, action_batch, reduction="none")
                * actor_weights
            ).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            loss_value = float((q_loss + value_loss + actor_loss).detach().cpu())

        if update % 100 == 0:
            target_network.load_state_dict(q_network.state_dict())
        if update % log_every == 0 or update == timesteps:
            eval_env = DataApiRoutingEnv(split="val", seed=seed + 1000)
            mean_reward = _evaluate_offline_policy(
                eval_env, q_network, actor, episodes=2, seed=seed + update
            )
            eval_env.close()
            row = {
                "step": update,
                "loss": loss_value,
                "mean_reward": mean_reward,
            }
            history.append(row)
            with (version_dir / "curve.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(row) + "\n")
            torch.save(
                {
                    "version": version,
                    "seed": seed,
                    "observation_size": states.shape[1],
                    "action_count": action_count,
                    "q_network": q_network.state_dict(),
                    "actor": actor.state_dict() if actor is not None else None,
                },
                checkpoint_dir / f"model_{update}_steps.pt",
            )

    metrics = {
        "version": version,
        "seed": seed,
        "timesteps": timesteps,
        "updates": timesteps,
        "offline_samples": len(states),
        "offline_behavior": "50% static candidate-0 policy, 50% random policy",
    }
    (version_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def _plot_version(version: str, results_root: Path, seeds: list[int]):
    curves = []
    for seed in seeds:
        path = results_root / version / f"seed_{seed}" / "curve.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            continue
        x = np.asarray([row["step"] for row in rows], dtype=float)
        y = np.asarray([row["mean_reward"] for row in rows], dtype=float)
        curves.append((x, y, seed))
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for x, y, seed in curves:
        ax.plot(x, y, label=f"seed={seed}")
    ax.set_xlabel("timesteps")
    ax.set_ylabel("mean reward")
    ax.set_title(f"{version} D2 training curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_root / version / "learning_curve.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the D2 agent variants for multiple seeds.")
    parser.add_argument("--versions", nargs="*", default=DEFAULT_VERSIONS)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--out-dir", type=Path, default=Path("results/d2_agents"))
    parser.add_argument("--evaluate-existing", action="store_true")
    args = parser.parse_args()

    results_root = args.out_dir
    results_root.mkdir(parents=True, exist_ok=True)

    if args.evaluate_existing:
        _evaluate_saved_ppo_models(results_root, args.versions, args.seeds)
        return

    for version in args.versions:
        version_root = results_root / version
        version_root.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            metrics_path = version_root / f"seed_{seed}" / "metrics.json"
            checkpoint_dir = version_root / f"seed_{seed}" / "checkpoints"
            if metrics_path.exists() and checkpoint_dir.exists():
                existing = json.loads(metrics_path.read_text(encoding="utf-8"))
                if int(existing.get("timesteps", -1)) == args.timesteps:
                    print(f"Skipping completed {version} seed={seed}")
                    continue
            if version in {"CQL", "IQL"}:
                _train_offline(version, seed, args.timesteps, results_root)
            else:
                _train_version(version, seed, args.timesteps, results_root)
        _plot_version(version, results_root, args.seeds)

    summary = {
        "versions": args.versions,
        "seeds": args.seeds,
        "timesteps": int(args.timesteps),
    }
    (results_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
