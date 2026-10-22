
"""Phase 7: linkage-seed and stale-information robustness tests."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.baselines.time_split import chronological_split

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "dataset" / "processed"
RESULTS = ROOT / "results"

SEEDS = [11, 22, 33, 44, 55]
DELAYS = [0, 1, 2, 5]


def linkage_sensitivity():
    """Summarize each linkage seed and compare its slot mappings."""
    rows = []
    mappings = {}

    for seed in SEEDS:
        folder = PROCESSED / f"linkage_seed_{seed}"
        meta_path = folder / "linkage_meta.json"
        mapping_path = folder / "slot_to_window.parquet"
        tx_path = folder / "linkage_tx.parquet"

        for path in [meta_path, mapping_path, tx_path]:
            if not path.exists():
                raise FileNotFoundError(f"Required Phase 7 input missing: {path}")

        meta = json.loads(meta_path.read_text())
        mapping = pd.read_parquet(mapping_path)
        mappings[seed] = mapping

        # Read only required columns from the large transaction file.
        tx = pd.read_parquet(
            tx_path,
            columns=[
                "e2e_ms_static",
                "sla_violation_static",
                "success_static",
            ],
        )

        rows.append({
            "seed": seed,
            "transactions": len(tx),
            "unique_windows": int(mapping["window_id"].nunique()),
            "mean_e2e_ms": float(tx["e2e_ms_static"].mean()),
            "p95_e2e_ms": float(tx["e2e_ms_static"].quantile(0.95)),
            "p99_e2e_ms": float(tx["e2e_ms_static"].quantile(0.99)),
            "sla_violation_rate": float(tx["sla_violation_static"].mean()),
            "success_rate": float(tx["success_static"].mean()),
            "metadata_sla_ms": meta["sla_ms"],
            "metadata_timeout_ms": meta["timeout_ms"],
        })
        print(f"Processed linkage seed {seed}")

    seed_summary = pd.DataFrame(rows)
    seed_summary.to_csv(
        RESULTS / "phase7_linkage_seed_sensitivity.csv", index=False
    )

    # Compare the mapping for the same minute slots across seed pairs.
    agreement_rows = []
    for a, b in combinations(SEEDS, 2):
        left = mappings[a].sort_values("slot").reset_index(drop=True)
        right = mappings[b].sort_values("slot").reset_index(drop=True)

        if not left["slot"].equals(right["slot"]):
            raise ValueError(f"Minute-slot alignment differs for seeds {a} and {b}")

        agreement_rows.append({
            "seed_a": a,
            "seed_b": b,
            "minute_slots": len(left),
            "mapping_agreement_rate": float(
                (left["window_id"].to_numpy()
                 == right["window_id"].to_numpy()).mean()
            ),
        })

    pd.DataFrame(agreement_rows).to_csv(
        RESULTS / "phase7_linkage_mapping_agreement.csv", index=False
    )
    return seed_summary, pd.DataFrame(agreement_rows)


def staleness_sensitivity():
    """
    Choose each stage's candidate using an earlier window's p95.
    Score that choice using the CURRENT window's p95 and SLA flag.
    """
    latency_path = PROCESSED / "stage_latency_windows.parquet"
    data = pd.read_parquet(latency_path)

    required = {
        "stage", "window_id", "cand_rank", "p95", "sla_violation"
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = data.copy()
    for col in ["window_id", "cand_rank", "p95", "sla_violation"]:
        data[col] = pd.to_numeric(data[col], errors="raise")

    data = data.dropna(subset=["stage", "window_id", "cand_rank", "p95"])
    data["window_id"] = data["window_id"].astype(int)
    data["cand_rank"] = data["cand_rank"].astype(int)
    data = data.sort_values(["stage", "window_id", "cand_rank"])

    # Use the project's established chronological split.
    train, validation, test = chronological_split(data)
    test_ids = sorted(test["window_id"].unique())
    if not test_ids:
        raise ValueError("Chronological test split is empty.")

    # Each (stage, window, candidate) must identify one row.
    if data.duplicated(["stage", "window_id", "cand_rank"]).any():
        raise ValueError("Duplicate stage/window/candidate rows found.")

    indexed = data.set_index(["stage", "window_id", "cand_rank"])
    stages = sorted(data["stage"].unique())
    candidates = sorted(data["cand_rank"].unique())
    rows = []

    for delay in DELAYS:
        for current_window in test_ids:
            observed_window = current_window - delay
            stage_latencies = []
            stage_sla_flags = []
            selected_ranks = []

            for stage in stages:
                observed = data[
                    (data["stage"] == stage)
                    & (data["window_id"] == observed_window)
                ]

                # For delay 0 this is a fresh-information benchmark.
                # For delays > 0, only the prior window is used to choose.
                if observed.empty:
                    raise ValueError(
                        f"No history for stage={stage}, "
                        f"window={observed_window}, delay={delay}"
                    )

                chosen = int(
                    observed.sort_values(["p95", "cand_rank"])
                    .iloc[0]["cand_rank"]
                )

                key = (stage, current_window, chosen)
                if key not in indexed.index:
                    raise ValueError(f"Missing current outcome for {key}")

                outcome = indexed.loc[key]
                stage_latencies.append(float(outcome["p95"]))
                stage_sla_flags.append(float(outcome["sla_violation"]))
                selected_ranks.append(chosen)

            rows.append({
                "delay_windows": delay,
                "current_window": int(current_window),
                "observed_window": int(current_window - delay),
                "stages": len(stages),
                "selected_candidate_ranks": ",".join(map(str, selected_ranks)),
                "sum_stage_p95_ms": float(sum(stage_latencies)),
                "mean_stage_p95_ms": float(np.mean(stage_latencies)),
                "stage_sla_violation_fraction": float(np.mean(stage_sla_flags)),
                "stage_sla_violation_count": int(sum(stage_sla_flags)),
            })

    detailed = pd.DataFrame(rows)
    detailed.to_csv(RESULTS / "phase7_staleness_detailed.csv", index=False)

    summary = (
        detailed.groupby("delay_windows")
        .agg(
            test_windows=("current_window", "nunique"),
            mean_sum_stage_p95_ms=("sum_stage_p95_ms", "mean"),
            mean_stage_p95_ms=("mean_stage_p95_ms", "mean"),
            mean_stage_sla_violation_fraction=(
                "stage_sla_violation_fraction", "mean"
            ),
            total_stage_sla_violations=("stage_sla_violation_count", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(RESULTS / "phase7_staleness_summary.csv", index=False)
    return detailed, summary


def write_report(linkage, agreement, stale):
    report = f"""# Phase 7 — Robustness and Sensitivity Evaluation

## 1. Linkage-seed sensitivity

Five linkage seeds were examined: {SEEDS}.

The transaction-to-window mapping agreement rates for all ten seed pairs
are saved in `phase7_linkage_mapping_agreement.csv`. Static-policy outcome
summaries are saved in `phase7_linkage_seed_sensitivity.csv`.

These runs use the project's synthetic PaySim-to-Alibaba linkage. They
measure sensitivity to the linkage random seed; they do not establish
that candidate 0 is representative of every alternative routing policy.

## 2. Staleness sensitivity

The evaluator compares information delays of {DELAYS} windows. Candidate
selection uses the stage p95 from the observed window; scoring uses the
selected candidate's p95 and SLA flag in the current window.

The delay-0 result is a **fresh-information benchmark** and is optimistic:
it uses current-window summaries to choose the candidate. It should not be
interpreted as a deployable policy result unless those measurements are
available before the routing decision.

Results are in:
- `phase7_staleness_detailed.csv`
- `phase7_staleness_summary.csv`

The summed stage p95 is a comparison proxy, not a measured end-to-end p95.
Percentiles from different stages cannot generally be added to obtain an
end-to-end percentile. The stage SLA violation fraction is the fraction
of selected stage-window records flagged as violations.

## 3. Limitations

- The chronological held-out test contains {stale["test_windows"].min()}
  windows per delay setting.
- Linkage outcomes are generated under a static candidate-0 policy.
- The staleness experiment is a candidate-selection sensitivity test,
  not a full online deployment replay.
- Confidence claims should be limited by the small number of test windows.
- AWS/SageMaker deployment remains pending.

## 4. Reproducibility

Run from the repository root:

`python -m dataset.baselines.phase7_robustness`

Generated results are CSV files under `results/`.
"""
    path = RESULTS / "phase7_robustness_report.md"
    path.write_text(report)
    print(f"\nReport written: {path}")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("=== Linkage-seed sensitivity ===")
    linkage, agreement = linkage_sensitivity()
    print(linkage.round(4).to_string(index=False))
    print("\nMapping agreement:")
    print(agreement.round(4).to_string(index=False))

    print("\n=== Staleness sensitivity ===")
    _, stale = staleness_sensitivity()
    print(stale.round(4).to_string(index=False))

    write_report(linkage, agreement, stale)
    print("\nPhase 7 finished.")


if __name__ == "__main__":
    main()
