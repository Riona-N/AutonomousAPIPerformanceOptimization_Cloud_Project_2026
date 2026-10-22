"""Train an A2 PPO adversary against a frozen routing PPO policy."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from attacks.adversary import LearnedAdversaryEnv
from src.ml_model.data_api_env import DataApiRoutingEnv


def train_adversary(
    router_path: Path,
    output_path: Path,
    timesteps: int = 50_000,
    epsilon: float = 0.1,
    seed: int = 42,
    split: str = "train",
    rollout_steps: int = 256,
) -> PPO:
    if timesteps <= 0:
        raise ValueError("timesteps must be greater than zero")
    if rollout_steps < 2:
        raise ValueError("rollout_steps must be at least 2")
    if epsilon <= 0:
        raise ValueError("epsilon must be greater than zero")

    router = PPO.load(str(router_path), device="auto")
    routing_env = DataApiRoutingEnv(split=split, seed=seed)
    adversary_env = LearnedAdversaryEnv(routing_env, router, epsilon=epsilon)
    n_steps = min(rollout_steps, max(2, timesteps))
    batch_size = min(64, n_steps)
    adversary = PPO(
        "MlpPolicy",
        adversary_env,
        seed=seed,
        n_steps=n_steps,
        batch_size=batch_size,
        verbose=1,
        device="auto",
    )
    adversary.learn(total_timesteps=timesteps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adversary.save(str(output_path))
    adversary_env.close()
    return adversary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-path", type=Path, default=Path("results/ppo_agent.zip"))
    parser.add_argument(
        "--output-path", type=Path, default=Path("results/adversary_ppo")
    )
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="train")
    parser.add_argument("--rollout-steps", type=int, default=256)
    args = parser.parse_args()

    if not args.router_path.exists():
        parser.error(f"Frozen router model not found: {args.router_path}")
    train_adversary(
        router_path=args.router_path,
        output_path=args.output_path,
        timesteps=args.timesteps,
        epsilon=args.epsilon,
        seed=args.seed,
        split=args.split,
        rollout_steps=args.rollout_steps,
    )


if __name__ == "__main__":
    main()
