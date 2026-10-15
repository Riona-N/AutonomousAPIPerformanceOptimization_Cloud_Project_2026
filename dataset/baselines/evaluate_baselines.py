from time_split import chronological_split
from pathlib import Path

import numpy as np
import pandas as pd

from rule_based import build_rule_policy, choose_candidate
from bandit import EpsilonGreedyBandit, UCB1, ThompsonSampling


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"
RESULTS_DIR = BASE_DIR.parent / "results"

WINDOWS_FILE = PROCESSED_DIR / "stage_latency_windows.parquet"
CANDIDATES_FILE = PROCESSED_DIR / "stage_candidates.parquet"

# Known stage SLAs, in milliseconds
STAGE_SLA_MS = {
    "auth": 625.0,
    "db": 2.0,
    "ingress": 152.0,
    "processor": 37.0,
    "routing": 689.45,
}

SEEDS = [11, 22, 33, 44, 55]
EPSILONS = [0.05, 0.1, 0.2]


# --------------------------------------------------
# LOAD AND PREPARE DATA
# --------------------------------------------------

def load_data():
    print("Loading processed Alibaba data...")

    windows = pd.read_parquet(WINDOWS_FILE)
    candidates = pd.read_parquet(CANDIDATES_FILE)

    required_window_columns = {
        "stage", "window_id", "cand_rank",
        "p95", "sla_violation"
    }

    missing = required_window_columns - set(windows.columns)

    if missing:
        raise ValueError(
            f"Missing columns in stage_latency_windows.parquet: {missing}"
        )

    print(f"Latency-window rows: {len(windows)}")
    print(f"Candidate-summary rows: {len(candidates)}")

    return windows, candidates


def prepare_windows(windows):
    windows = windows.copy()

    windows["window_id"] = pd.to_numeric(
        windows["window_id"], errors="raise"
    )

    windows["cand_rank"] = pd.to_numeric(
        windows["cand_rank"], errors="raise"
    )

    windows["p95"] = pd.to_numeric(
        windows["p95"], errors="coerce"
    )

    windows = windows.dropna(
        subset=["stage", "window_id", "cand_rank", "p95"]
    )

    windows = windows.sort_values(
        ["stage", "window_id", "cand_rank"]
    ).reset_index(drop=True)

    return windows


# --------------------------------------------------
# REWARD FOR LEARNING
# --------------------------------------------------

def learning_reward(stage, latency_ms):
    """
    Convert latency into a reward between 0 and 1.

    Lower latency produces a higher reward.
    The stage SLA provides the normalization scale.
    """

    scale = STAGE_SLA_MS.get(stage)

    if scale is None or scale <= 0:
        raise ValueError(f"Missing or invalid SLA for stage: {stage}")

    return 1.0 / (1.0 + max(latency_ms, 0.0) / scale)


# --------------------------------------------------
# RECORD THE SELECTED CANDIDATE
# --------------------------------------------------

def make_result(method, stage, window_id, candidate, selected, seed=None):
    latency = float(selected["p95"])
    sla_violation = int(selected["sla_violation"])

    return {
        "method": method,
        "seed": seed,
        "stage": stage,
        "window_id": window_id,
        "candidate": int(candidate),
        "latency_ms": latency,
        "sla_violation": sla_violation,
        "reward": -latency,
    }


# --------------------------------------------------
# RULE-BASED BASELINE
# --------------------------------------------------

def evaluate_rule_based(train_windows, test_windows, seed=None):
    print("Running rule-based baseline...")

    # Learn the best candidate using training windows only.
    training_summary = (
        train_windows.groupby(["stage", "cand_rank"])["p95"]
        .mean()
        .reset_index()
    )

    best_candidates = (
        training_summary.sort_values(
            ["stage", "p95", "cand_rank"]
        )
        .drop_duplicates(subset=["stage"])
        .set_index("stage")["cand_rank"]
        .to_dict()
    )

    results = []

    # Evaluate the fixed policy on test windows only.
    for stage in sorted(test_windows["stage"].unique()):
        stage_data = test_windows[
            test_windows["stage"] == stage
        ]

        candidate = int(best_candidates[stage])

        for window_id in sorted(stage_data["window_id"].unique()):
            window_data = stage_data[
                stage_data["window_id"] == window_id
            ]

            selected = window_data[
                window_data["cand_rank"] == candidate
            ]

            if selected.empty:
                continue

            results.append(
                make_result(
                    "rule_based",
                    stage,
                    window_id,
                    candidate,
                    selected.iloc[0],
                    seed=seed,
                )
            )

    return pd.DataFrame(results)


# --------------------------------------------------
# GENERIC BANDIT EVALUATOR
# --------------------------------------------------

def evaluate_bandit_policy(windows, method, seed, epsilon=None):
    """
    Run one learning policy over chronological windows.

    Each stage has its own bandit. The policy sees the
    selected candidate's latency only after making its choice.
    """

    results = []

    for stage in sorted(windows["stage"].unique()):
        stage_data = windows[
            windows["stage"] == stage
        ].copy()

        stage_data = stage_data.sort_values("window_id")

        available_candidates = sorted(
            stage_data["cand_rank"].unique()
        )

        if len(available_candidates) == 0:
            continue

        # Candidate ranks must be consecutive: 0, 1, 2, ...
        expected_candidates = list(range(len(available_candidates)))

        if available_candidates != expected_candidates:
            raise ValueError(
                f"{stage}: expected candidate ranks "
                f"{expected_candidates}, found {available_candidates}"
            )

        n_candidates = len(available_candidates)

        if method == "epsilon_greedy":
            bandit = EpsilonGreedyBandit(
                n_candidates=n_candidates,
                epsilon=epsilon,
                seed=seed,
            )

            method_name = f"epsilon_greedy_{epsilon}"

        elif method == "ucb1":
            bandit = UCB1(n_candidates=n_candidates)
            method_name = "ucb1"

        elif method == "thompson_sampling":
            bandit = ThompsonSampling(
                n_candidates=n_candidates,
                seed=seed,
            )

            method_name = "thompson_sampling"

        else:
            raise ValueError(f"Unknown bandit method: {method}")

        for window_id in sorted(stage_data["window_id"].unique()):
            window_data = stage_data[
                stage_data["window_id"] == window_id
            ]

            candidate = bandit.choose()

            selected = window_data[
                window_data["cand_rank"] == candidate
            ]

            if selected.empty:
                # Do not update the bandit if the chosen
                # candidate has no observation for this window.
                continue

            selected_row = selected.iloc[0]
            latency = float(selected_row["p95"])

            # Bounded reward for bandit learning
            reward_for_learning = learning_reward(stage, latency)

            bandit.update(candidate, reward_for_learning)

            results.append(
                make_result(
                    method_name,
                    stage,
                    window_id,
                    candidate,
                    selected_row,
                    seed=seed,
                )
            )

    return pd.DataFrame(results)


# --------------------------------------------------
# CALCULATE SUMMARY METRICS
# --------------------------------------------------

def calculate_metrics(results):
    summary = []

    for method, group in results.groupby("method"):
        latencies = group["latency_ms"]

        summary.append({
            "method": method,
            "average_latency_ms": latencies.mean(),
            "p95_latency_ms": latencies.quantile(0.95),
            "p99_latency_ms": latencies.quantile(0.99),
            "sla_violation_rate": group["sla_violation"].mean(),
            "average_reward": group["reward"].mean(),
            "windows_evaluated": len(group),
            "seeds_evaluated": group["seed"].nunique(),
        })

    return pd.DataFrame(summary).sort_values("method")


# --------------------------------------------------
# MAIN
# --------------------------------------------------


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    windows, candidates = load_data()
    windows = prepare_windows(windows)

    # Split windows chronologically
    train, validation, test = chronological_split(windows)

    print("\nCHRONOLOGICAL DATA SPLIT")
    print(f"Training windows: {sorted(train['window_id'].unique())}")
    print(f"Validation windows: {sorted(validation['window_id'].unique())}")
    print(f"Test windows: {sorted(test['window_id'].unique())}")

    # For this first integration, evaluate the policies on the
    # held-out test windows. Bandit warm-up/learning is handled
    # separately in the next step.
    rule_results = evaluate_rule_based(test, candidates)

    all_results = [rule_results]

    for seed in SEEDS:
        print(f"Evaluating test windows with seed {seed}...")

        for epsilon in EPSILONS:
            result = evaluate_bandit_policy(
                test,
                method="epsilon_greedy",
                seed=seed,
                epsilon=epsilon,
            )
            all_results.append(result)

        for method in ["ucb1", "thompson_sampling"]:
            result = evaluate_bandit_policy(
                test,
                method=method,
                seed=seed,
            )
            all_results.append(result)

    detailed = pd.concat(all_results, ignore_index=True)
    summary = calculate_metrics(detailed)

    detailed_path = RESULTS_DIR / "baseline_test_detailed_results.csv"
    summary_path = RESULTS_DIR / "baseline_test_comparison.csv"

    detailed.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nHELD-OUT TEST RESULTS")
    print("=" * 75)
    print(summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))

    print("\nSaved detailed test results to:")
    print(detailed_path)

    print("\nSaved test summary to:")
    print(summary_path)


if __name__ == "__main__":
    main()