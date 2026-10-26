"""Train and compare D4 reward-weight combinations on validation data only."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from src.ml_model.data_api_env import DataApiRoutingEnv


def evaluate(
    model: PPO,
    env: DataApiRoutingEnv,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    returns = []
    component_totals = {
        "base_reward": [],
        "cost": [],
        "low_trust_fraction": [],
        "normalized_fraud_value": [],
        "sla_violation": [],
        "success": [],
    }
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total_return = 0.0
        episode_totals = {key: 0.0 for key in component_totals}
        steps = 0
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            terms = info["reward_terms"]
            episode_totals["base_reward"] += terms["base_reward"]
            episode_totals["cost"] += terms["cost"]
            episode_totals["low_trust_fraction"] += terms["low_trust_fraction"]
            episode_totals["normalized_fraud_value"] += terms["normalized_fraud_value"]
            episode_totals["sla_violation"] += float(info["sla_violation"])
            episode_totals["success"] += float(info["success"])
            steps += 1
            if terminated or truncated:
                break
        returns.append(total_return)
        for key in episode_totals:
            component_totals[key].append(episode_totals[key] / max(steps, 1))

    means = {f"mean_{key}": float(np.mean(values)) for key, values in component_totals.items()}
    means["mean_episode_return"] = float(np.mean(returns))
    means["std_episode_return"] = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    means["selection_score"] = (
        means["mean_base_reward"]
        - means["mean_low_trust_fraction"]
        - means["mean_normalized_fraud_value"]
    )
    return means


def run_grid_search(
    output_dir: Path,
    timesteps: int = 2048,
    seeds: tuple[int, ...] = (42,),
    episodes: int = 3,
    beta_values: tuple[float, ...] = (0.0, 0.1, 1.0),
    gamma_values: tuple[float, ...] = (0.0, 0.5, 1.0),
    delta_values: tuple[float, ...] = (0.0, 0.5, 1.0),
) -> list[dict[str, float]]:
    if timesteps <= 0 or episodes <= 0 or not seeds:
        raise ValueError("timesteps, episodes and seeds must be positive")
    if any(value < 0 for value in beta_values + gamma_values + delta_values):
        raise ValueError("grid values must be non-negative")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []
    best_score = float("-inf")
    best_model_path: Path | None = None
    combinations = list(itertools.product(beta_values, gamma_values, delta_values))

    for beta, gamma, delta in combinations:
        for seed in seeds:
            run_name = f"beta_{beta:g}_gamma_{gamma:g}_delta_{delta:g}_seed_{seed}"
            run_dir = output_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            train_env = DataApiRoutingEnv(
                split="train",
                seed=seed,
                reward_beta=beta,
                reward_gamma=gamma,
                reward_delta=delta,
            )
            model = PPO(
                "MlpPolicy",
                train_env,
                seed=seed,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                gamma=0.99,
                ent_coef=0.01,
                verbose=0,
                device="auto",
            )
            model.learn(total_timesteps=timesteps)
            model_path = run_dir / "model"
            model.save(str(model_path))
            train_env.close()

            val_env = DataApiRoutingEnv(
                split="val",
                seed=seed + 10_000,
                reward_beta=beta,
                reward_gamma=gamma,
                reward_delta=delta,
            )
            metrics = evaluate(model, val_env, episodes, seed + 20_000)
            val_env.close()
            row = {
                "beta": beta,
                "gamma": gamma,
                "delta": delta,
                "seed": seed,
                "timesteps": int(model.num_timesteps),
                **metrics,
            }
            rows.append(row)
            (run_dir / "validation_metrics.json").write_text(
                json.dumps(row, indent=2), encoding="utf-8"
            )
            if metrics["selection_score"] > best_score:
                best_score = metrics["selection_score"]
                best_model_path = model_path.with_suffix(".zip")
            del model

    csv_path = output_dir / "grid_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    beta_identifiable = any(row["mean_cost"] > 0.0 for row in rows)
    gamma_identifiable = any(row["mean_low_trust_fraction"] > 0.0 for row in rows)
    delta_identifiable = any(
        row["mean_normalized_fraud_value"] > 0.0 for row in rows
    )
    tuning_identifiable = beta_identifiable or gamma_identifiable or delta_identifiable
    selected = max(rows, key=lambda row: row["selection_score"])
    reference_beta, reference_gamma, reference_delta = combinations[0]
    reference_seed = seeds[0]
    summary = {
        "split_used_for_tuning": "val",
        "training_split": "train",
        "grid_combinations": len(combinations),
        "seeds": list(seeds),
        "timesteps_requested": timesteps,
        "timesteps_per_run_actual": sorted({int(row["timesteps"]) for row in rows}),
        "selected_weights": (
            {
                "beta": selected["beta"],
                "gamma": selected["gamma"],
                "delta": selected["delta"],
            }
            if tuning_identifiable
            else None
        ),
        "weight_identifiability": {
            "beta": beta_identifiable,
            "gamma": gamma_identifiable,
            "delta": delta_identifiable,
        },
        "tuning_status": (
            "validation_grid_completed"
            if tuning_identifiable
            else "not_identifiable_all_penalty_signals_zero"
        ),
        "selection_score_definition": (
            "mean base reward - mean low-trust fraction - mean normalized fraud value; "
            "validation-only, fixed unit weights for model selection"
        ),
        "best_model_path": (
            str(best_model_path) if tuning_identifiable and best_model_path else None
        ),
        "reference_model_path": str(
            output_dir
            / (
                f"beta_{reference_beta:g}_gamma_{reference_gamma:g}_"
                f"delta_{reference_delta:g}_seed_{reference_seed}"
            )
            / "model.zip"
        ),
        "cost_data_available": False,
        "beta_identifiable": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [
        f"β={row['beta']:g}, γ={row['gamma']:g}, δ={row['delta']:g}\nseed={row['seed']}"
        for row in rows
    ]
    ax.bar(np.arange(len(rows)), [row["selection_score"] for row in rows])
    ax.set_xticks(np.arange(len(rows)), labels, rotation=90, fontsize=7)
    ax.set_ylabel("validation selection score")
    ax.set_title("D4 validation grid search")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "grid_search.png", dpi=200)
    plt.close(fig)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=2048)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("results/d4_grid"))
    parser.add_argument("--beta", nargs="*", type=float, default=[0.0, 0.1, 1.0])
    parser.add_argument("--gamma", nargs="*", type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument("--delta", nargs="*", type=float, default=[0.0, 0.5, 1.0])
    args = parser.parse_args()
    run_grid_search(
        output_dir=args.out_dir,
        timesteps=args.timesteps,
        seeds=tuple(args.seeds),
        episodes=args.episodes,
        beta_values=tuple(args.beta),
        gamma_values=tuple(args.gamma),
        delta_values=tuple(args.delta),
    )


if __name__ == "__main__":
    main()
