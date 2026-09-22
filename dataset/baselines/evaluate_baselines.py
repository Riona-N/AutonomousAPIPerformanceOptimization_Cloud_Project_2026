from pathlib import Path

import pandas as pd

from rule_based import build_rule_policy, choose_candidate
from bandit import EpsilonGreedyBandit


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "processed"

RESULTS_DIR = BASE_DIR.parent / "results"

WINDOWS_FILE = PROCESSED_DIR / "stage_latency_windows.parquet"
CANDIDATES_FILE = PROCESSED_DIR / "stage_candidates.parquet"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

def load_data():

    print("Loading Alibaba processed data...")

    windows = pd.read_parquet(WINDOWS_FILE)
    candidates = pd.read_parquet(CANDIDATES_FILE)

    print(f"Latency windows loaded: {len(windows)} rows")
    print(f"Candidates loaded: {len(candidates)} rows")

    return windows, candidates


# ---------------------------------------------------------
# Prepare evaluation windows
# ---------------------------------------------------------

def prepare_windows(windows):

    windows = windows.sort_values(
        ["stage", "window_id"]
    ).reset_index(drop=True)

    return windows


# ---------------------------------------------------------
# Static rule-based baseline
# ---------------------------------------------------------

def evaluate_rule_based(windows, candidates):

    print("\nRunning rule-based baseline...")

    policy = build_rule_policy(candidates)

    print("\nStatic routing policy:")

    for stage in sorted(policy):

        print(
            f"  {stage} -> candidate {policy[stage]}"
        )

    results = []

    unique_windows = (
        windows[
            ["stage", "window_id"]
        ]
        .drop_duplicates()
        .sort_values(["stage", "window_id"])
    )

    for _, row in unique_windows.iterrows():

        stage = row["stage"]
        window_id = row["window_id"]

        candidate = choose_candidate(
            policy,
            stage
        )

        selected = windows[
            (windows["stage"] == stage)
            & (windows["window_id"] == window_id)
            & (windows["cand_rank"] == candidate)
        ]

        if selected.empty:
            continue

        selected = selected.iloc[0]

        latency = float(selected["p95"])
        results.append({
            "method": "rule_based",
            "stage": stage,
            "window_id": window_id,
            "candidate": candidate,
            "latency_ms": latency,
            "sla_violation": int(selected["sla_violation"]),
    "reward": -latency
})

    return pd.DataFrame(results)


# ---------------------------------------------------------
# Bandit baseline
# ---------------------------------------------------------

def evaluate_bandit(windows):

    print("\nRunning epsilon-greedy bandit...")

    results = []

    stages = sorted(
        windows["stage"].unique()
    )

    for stage in stages:

        stage_data = windows[
            windows["stage"] == stage
        ].copy()

        stage_data = stage_data.sort_values(
            "window_id"
        )

        bandit = EpsilonGreedyBandit(
            n_candidates=3,
            epsilon=0.1,
            seed=42
        )

        window_ids = (
            stage_data["window_id"]
            .drop_duplicates()
            .sort_values()
        )

        for window_id in window_ids:

            window_data = stage_data[
                stage_data["window_id"] == window_id
            ]

            if window_data.empty:
                continue

            candidate = bandit.choose()

            selected = window_data[
                window_data["cand_rank"] == candidate
            ]

            if selected.empty:
                continue

            selected = selected.iloc[0]

            latency = float(
                selected["p95"]
            )

            sla_violation = int(
                selected["sla_violation"]
            )

            # Lower latency = better reward
            reward = -latency

            bandit.update(
                candidate,
                reward
            )

            results.append({
                "method": "epsilon_greedy_bandit",
                "stage": stage,
                "window_id": window_id,
                "candidate": candidate,
                "latency_ms": latency,
                "sla_violation": sla_violation,
                "reward": reward
            })

        print(
            f"  {stage}: "
            f"candidate selections = "
            f"{bandit.get_counts()}"
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------
# Calculate metrics
# ---------------------------------------------------------

def calculate_metrics(results):

    summary = []

    for method in results["method"].unique():

        method_data = results[
            results["method"] == method
        ]

        average_latency = (
            method_data["latency_ms"].mean()
        )

        p95_latency = (
            method_data["latency_ms"].quantile(0.95)
        )

        sla_rate = (
            method_data["sla_violation"].mean()
        )

        total_windows = len(method_data)

        average_reward = method_data["reward"].mean()
        
        summary.append({
            "method": method,
            "average_latency_ms": average_latency,
            "p95_latency_ms": p95_latency,
            "sla_violation_rate": sla_rate,
            "average_reward": average_reward,
            "windows_evaluated": total_windows
        })

    return pd.DataFrame(summary)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    windows, candidates = load_data()

    windows = prepare_windows(windows)

    # -----------------------------
    # Rule-based
    # -----------------------------

    rule_results = evaluate_rule_based(
        windows,
        candidates
    )

    # -----------------------------
    # Bandit
    # -----------------------------

    bandit_results = evaluate_bandit(
        windows
    )

    # -----------------------------
    # Combine results
    # -----------------------------

    all_results = pd.concat(
        [
            rule_results,
            bandit_results
        ],
        ignore_index=True
    )

    # -----------------------------
    # Metrics
    # -----------------------------

    summary = calculate_metrics(
        all_results
    )

    # -----------------------------
    # Save detailed results
    # -----------------------------

    all_results.to_csv(
        RESULTS_DIR / "baseline_detailed_results.csv",
        index=False
    )

    summary.to_csv(
        RESULTS_DIR / "baseline_comparison.csv",
        index=False
    )

    # -----------------------------
    # Print results
    # -----------------------------

    print("\n")
    print("=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nResults saved to:")

    print(
        RESULTS_DIR /
        "baseline_detailed_results.csv"
    )

    print(
        RESULTS_DIR /
        "baseline_comparison.csv"
    )


if __name__ == "__main__":
    main()