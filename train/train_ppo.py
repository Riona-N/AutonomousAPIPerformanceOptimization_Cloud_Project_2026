"""Train and evaluate a PPO routing agent with reproducible logging and checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from src.ml_model.data_api_env import DataApiRoutingEnv


def evaluate(model: PPO, episodes: int, seed: int) -> dict[str, float]:
    rewards: list[float] = []
    latencies: list[float] = []
    successes: list[float] = []
    sla_violations: list[float] = []

    for episode in range(episodes):
        env = DataApiRoutingEnv(split="val", seed=seed + episode)
        observation, _ = env.reset(seed=seed + episode)
        episode_reward = 0.0
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            latencies.append(float(info["e2e_ms"]))
            successes.append(float(info["success"]))
            sla_violations.append(float(info["sla_violation"]))
            if terminated or truncated:
                break
        rewards.append(episode_reward)

    return {
        "episodes": float(episodes),
        "mean_episode_reward": float(np.mean(rewards)),
        "mean_e2e_ms": float(np.mean(latencies)),
        "success_rate": float(np.mean(successes)),
        "sla_violation_rate": float(np.mean(sla_violations)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=Path, default=Path("results/ppo_agent"))
    parser.add_argument("--metrics-path", type=Path, default=Path("results/ppo_metrics.json"))
    parser.add_argument("--tensorboard-log", type=Path, default=Path("results/tensorboard/ppo"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/checkpoints/ppo"))
    parser.add_argument("--checkpoint-freq", type=int, default=500)
    args = parser.parse_args()

    np.random.seed(args.seed)

    env = DataApiRoutingEnv(split="train", seed=args.seed)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        n_steps=256,
        batch_size=64,
        tensorboard_log=str(args.tensorboard_log),
        device="auto",
    )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.tensorboard_log.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq),
        save_path=str(args.checkpoint_dir),
        name_prefix="ppo",
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
    model.save(str(args.model_path))

    metrics = evaluate(model, episodes=args.episodes, seed=args.seed + 10_000)
    args.metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
