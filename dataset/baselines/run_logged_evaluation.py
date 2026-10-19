"""Export Role 3 decision logs and offline-RL transition proxies.

Run from the repository root:
    python -m dataset.baselines.run_logged_evaluation

This invokes the established evaluator, then adds three timestamp-matched
artifacts. The transition data is an offline proxy, not live/on-policy RL data.
"""
import json
import re
from datetime import datetime, timezone

import pandas as pd
from dataset.baselines import evaluate_baselines as evaluator
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
WINDOWS_PATH = ROOT / "dataset" / "processed" / "stage_latency_windows.parquet"
FEATURES = ("n_calls", "mean_ms", "p50", "p95", "p99")


def latest_detailed():
    files = list(RESULTS.glob("baseline_test_detailed_results_*.csv"))
    if not files:
        raise FileNotFoundError("No detailed evaluator CSV found under results/.")
    return max(files, key=lambda p: p.stat().st_mtime)


def run_id_from(path):
    import os
    import re

    name = os.path.basename(path)
    match = re.search(r"baseline_test_detailed_results_(.+)\.csv$", name)
    if match:
        return match.group(1)
    return "unknown_run"



def export(detailed_path):
    detail = pd.read_csv(detailed_path)
    required = {"method", "seed", "stage", "window_id", "candidate",
                "latency_ms", "sla_violation", "reward"}
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"Detailed results missing columns: {sorted(missing)}")

    windows = pd.read_parquet(WINDOWS_PATH)
    features = [f for f in FEATURES if f in windows.columns]
    if not features:
        raise ValueError(f"No supported state features found. Expected one of {FEATURES}.")
    context_frame = (windows.groupby(["stage", "window_id"], as_index=False)[features]
                     .mean(numeric_only=True).sort_values(["stage", "window_id"]))
    context = {}
    ordered = {}
    for stage, group in context_frame.groupby("stage"):
        stage = str(stage)
        ordered[stage] = [int(x) for x in group["window_id"].tolist()]
        for _, row in group.iterrows():
            context[(stage, int(row["window_id"]))] = {
                f: float(row[f]) for f in features
            }

    run_id = run_id_from(detailed_path)
    created = datetime.now(timezone.utc).isoformat()
    logs = detail.copy()
    logs.insert(0, "run_id", run_id)
    logs.insert(1, "logged_at_utc", created)
    logs.insert(2, "event_type", "routing_decision")
    logs["action"] = logs["candidate"]
    logs["observed_latency_ms"] = logs["latency_ms"]
    logs["legacy_reward"] = logs["reward"]
    logs["bounded_reward"] = logs.apply(
        lambda r: 1.0 / (1.0 + max(float(r["latency_ms"]), 0.0) /
                         float(evaluator.STAGE_SLA_MS[str(r["stage"])])), axis=1)
    logs["is_oracle_reference"] = logs["method"].astype(str).str.lower().eq("oracle")
    log_path = RESULTS / f"baseline_decision_logs_{run_id}.csv"
    logs.to_csv(log_path, index=False)

    transitions = []
    for row in detail.itertuples(index=False):
        method, stage, wid = str(row.method), str(row.stage), int(row.window_id)
        # Oracle uses observed outcomes to choose and is not suitable as a training policy.
        if method.lower() == "oracle":
            continue
        ids = ordered.get(stage, [])
        if wid not in ids:
            continue
        pos = ids.index(wid)
        if pos == 0:
            continue
        prev_id = ids[pos - 1]
        s, ns = context[(stage, prev_id)], context[(stage, wid)]
        sla = float(evaluator.STAGE_SLA_MS[stage])
        bounded = 1.0 / (1.0 + max(float(row.latency_ms), 0.0) / sla)
        item = {
            "run_id": run_id, "method": method, "seed": int(row.seed),
            "stage": stage, "state_window_id": prev_id, "window_id": wid,
            "action": int(row.candidate), "reward": bounded,
            "observed_latency_ms": float(row.latency_ms),
            "sla_violation": int(row.sla_violation),
            "next_state_window_id": wid, "done": int(pos == len(ids) - 1),
        }
        for f in features:
            item[f"state_{f}"] = s[f]
            item[f"next_state_{f}"] = ns[f]
        transitions.append(item)

    transition_path = RESULTS / f"offline_rl_transitions_{run_id}.csv"
    pd.DataFrame(transitions).to_csv(transition_path, index=False)
    schema = {
        "run_id": run_id, "created_at_utc": created,
        "decision_log_file": log_path.name, "transition_file": transition_path.name,
        "state_features": features,
        "reward_definition": "1 / (1 + max(latency_ms, 0) / stage_SLA_MS[stage]); higher is better.",
        "construction": "state is the previous chronological window's stage-level mean context; next state is the current window's mean context.",
        "limitations": [
            "Offline proxy from processed latency windows; not live on-policy experience.",
            "Aggregated candidate rows may include counterfactual information unavailable to a live router.",
            "Oracle is excluded from the trainable transition export.",
            "Only five test windows per stage are available in the current 30-window dataset."
        ]
    }
    schema_path = RESULTS / f"offline_rl_transition_schema_{run_id}.json"
    schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return log_path, transition_path, schema_path


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    evaluator.main()
    detailed_path = latest_detailed()
    log_path, transition_path, schema_path = export(detailed_path)
    print("\\nRole 3 exports created:")
    for path in (detailed_path, log_path, transition_path, schema_path):
        print(f"- {path.relative_to(ROOT)}")
    print("Review the schema limitations before describing these as offline-RL data.")


if __name__ == "__main__":
    main()
