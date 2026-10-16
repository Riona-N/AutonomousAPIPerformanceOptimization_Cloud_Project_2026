from pathlib import Path

import numpy as np
import pandas as pd

from dataset.baselines.time_split import chronological_split
from dataset.baselines.rule_based import build_rule_policy, choose_candidate
from dataset.baselines.bandit import EpsilonGreedyBandit, UCB1, ThompsonSampling
from dataset.baselines.additional_policies import (
    RandomRouting,
    LeastLoaded,
    PowerOfTwoChoices,
    LinUCB,
)

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

def evaluate_count_policy(windows, method, seed):
    """Evaluate random and count-based routing policies."""

    results = []
    policies = {}

    # Give each stage its own routing policy.
    for stage in windows["stage"].unique():
        if method == "random":
            policies[stage] = RandomRouting(seed=seed)

        elif method == "least_loaded":
            policies[stage] = LeastLoaded(seed=seed)

        elif method == "power_of_two_choices":
            policies[stage] = PowerOfTwoChoices(seed=seed)

        else:
            raise ValueError(f"Unknown policy: {method}")

    # Evaluate windows chronologically.
    ordered = windows.sort_values(["window_id", "stage"])

    for (stage, window_id), group in ordered.groupby(
        ["stage", "window_id"], sort=False
    ):
        policy = policies[stage]

        candidates = sorted(
            group["cand_rank"].astype(int).unique().tolist()
        )

        candidate = policy.choose(candidates)

        selected = group[
            group["cand_rank"].astype(int) == candidate
        ].iloc[0]

        latency = float(selected["p95"])

        results.append({
            "method": method,
            "seed": seed,
            "stage": stage,
            "window_id": window_id,
            "candidate": candidate,
            "latency_ms": latency,
            "sla_violation": int(selected["sla_violation"]),
            "reward": -latency,
        })

        # Update routing counts after making the selection.
        if hasattr(policy, "update"):
            policy.update(candidate)

    return pd.DataFrame(results)

def evaluate_linucb(train_windows, test_windows, seed):
    """Evaluate LinUCB using candidate features learned from training data."""

    feature_columns = [
        "n_calls",
        "mean_ms",
        "p50",
        "p95",
        "p99",
    ]

    # Candidate features are calculated from training data only.
    candidate_summary = (
        train_windows.groupby(["stage", "cand_rank"])[feature_columns]
        .mean()
        .reset_index()
    )

    policies = {}
    results = []

    for stage, group in candidate_summary.groupby("stage"):
        candidates = sorted(group["cand_rank"].astype(int).unique())

        values = group.set_index("cand_rank")[feature_columns]
        values = values.loc[candidates].astype(float)

        # Log-transform counts and standardize features within each stage.
        values["n_calls"] = np.log1p(values["n_calls"].clip(lower=0))

        means = values.mean()
        stds = values.std(ddof=0).replace(0, 1)
        values = (values - means) / stds

        # Add an intercept feature.
        values.insert(0, "intercept", 1.0)

        candidate_features = {
            candidate: values.loc[candidate].to_numpy(dtype=float)
            for candidate in candidates
        }

        policies[stage] = LinUCB(
            candidate_features=candidate_features,
            alpha=1.0,
        )

    # Only test-window outcomes are used as feedback after selection.
    ordered = test_windows.sort_values(["window_id", "stage"])

    for (stage, window_id), group in ordered.groupby(
        ["stage", "window_id"], sort=False
    ):
        policy = policies[stage]

        candidates = sorted(
            group["cand_rank"].astype(int).unique().tolist()
        )

        candidate = policy.choose(candidates)

        selected = group[
            group["cand_rank"].astype(int) == candidate
        ].iloc[0]

        latency = float(selected["p95"])
        reward = learning_reward(stage, latency)

        results.append({
            "method": "linucb",
            "seed": seed,
            "stage": stage,
            "window_id": window_id,
            "candidate": candidate,
            "latency_ms": latency,
            "sla_violation": int(selected["sla_violation"]),
            "reward": -latency,
        })

        policy.update(candidate, reward)

    return pd.DataFrame(results)


def evaluate_oracle(test_windows, seed):
    """
    Optimistic reference: select the candidate with the lowest
    observed P95 latency for each stage-window.

    This uses the outcome being evaluated, so it is not deployable.
    """

    results = []

    ordered = test_windows.sort_values(["window_id", "stage"])

    for (stage, window_id), group in ordered.groupby(
        ["stage", "window_id"], sort=False
    ):
        selected = group.loc[group["p95"].idxmin()]

        results.append({
            "method": "oracle",
            "seed": seed,
            "stage": stage,
            "window_id": window_id,
            "candidate": int(selected["cand_rank"]),
            "latency_ms": float(selected["p95"]),
            "sla_violation": int(selected["sla_violation"]),
            "reward": -float(selected["p95"]),
        })

    return pd.DataFrame(results)

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

    # Split the data in chronological order.
    train, validation, test = chronological_split(windows)

    print("\nCHRONOLOGICAL DATA SPLIT")
    print(f"Training windows: {sorted(train['window_id'].unique())}")
    print(f"Validation windows: {sorted(validation['window_id'].unique())}")
    print(f"Test windows: {sorted(test['window_id'].unique())}")

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Training, validation, and test splits must all contain data."
        )

    all_results = []

    # Evaluate every method with the required five seeds.
    for seed in SEEDS:
        print(f"\nEvaluating seed {seed}...")

        # 1. Rule-based baseline
        print("Running rule-based baseline...")
        rule_results = evaluate_rule_based(
            train_windows=train,
            test_windows=test,
            seed=seed,
        )
        all_results.append(rule_results)

        # 2. Epsilon-greedy baselines
        for epsilon in EPSILONS:
            print(f"Running epsilon-greedy (epsilon={epsilon})...")
            result = evaluate_bandit_policy(
                test,
                method="epsilon_greedy",
                seed=seed,
                epsilon=epsilon,
            )
            all_results.append(result)

        # 3. UCB1 and Thompson sampling
        for method in ["ucb1", "thompson_sampling"]:
            print(f"Running {method} baseline...")
            result = evaluate_bandit_policy(
                test,
                method=method,
                seed=seed,
            )
            all_results.append(result)

        # 4. Random, least-loaded, and power-of-two choices
        for method in [
            "random",
            "least_loaded",
            "power_of_two_choices",
        ]:
            print(f"Running {method} baseline...")
            result = evaluate_count_policy(
                windows=test,
                method=method,
                seed=seed,
            )
            all_results.append(result)

        # 5. LinUCB — exactly once per seed
        print("Running LinUCB baseline...")
        all_results.append(
            evaluate_linucb(
                train_windows=train,
                test_windows=test,
                seed=seed,
            )
        )

        # 6. Oracle — exactly once per seed
        print("Running oracle reference...")
        all_results.append(
            evaluate_oracle(
                test_windows=test,
                seed=seed,
            )
        )

    detailed = pd.concat(all_results, ignore_index=True)
    summary = calculate_metrics(detailed)

    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")

    detailed_path = RESULTS_DIR / f"baseline_test_detailed_results_{run_id}.csv"
    summary_path = RESULTS_DIR / f"baseline_test_comparison_{run_id}.csv"

    for path in [detailed_path, summary_path]:
        if path.exists():
            raise FileExistsError(
                f"{path} already exists. Preserve it and choose new "
                "output filenames before running again."
            )

    detailed.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nHELD-OUT TEST RESULTS")
    print("=" * 75)
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved detailed test results to:")
    print(detailed_path)

    print("\nSaved test summary to:")
    print(summary_path)


if __name__ == "__main__":
    main()