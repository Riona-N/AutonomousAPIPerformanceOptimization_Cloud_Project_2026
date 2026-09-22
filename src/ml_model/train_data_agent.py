"""Train and evaluate the Role 1 PPO baseline on Role 3's processed data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .data_api_env import DataApiRoutingEnv


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
            episode_reward += reward
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
    parser.add_argument("--output", type=Path, default=Path("results/data_ppo_metrics.json"))
    parser.add_argument("--model", type=Path, default=Path("results/data_ppo_agent"))
    args = parser.parse_args()

    env = DataApiRoutingEnv(split="train", seed=args.seed)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        n_steps=256,
        batch_size=64,
        device="auto",
    )
    model.learn(total_timesteps=args.timesteps)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.model))
    metrics = evaluate(model, episodes=args.episodes, seed=args.seed + 10_000)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()