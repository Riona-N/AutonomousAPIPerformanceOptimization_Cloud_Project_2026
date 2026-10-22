
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "data" / "offline.h5"

# Select the newest exported transition CSV.
files = list(RESULTS.glob("offline_rl_transitions_*.csv"))
if not files:
    raise FileNotFoundError("No offline RL transition CSV found in results/")

csv_path = max(files, key=lambda p: p.stat().st_mtime)
df = pd.read_csv(csv_path)

state_cols = [
    "state_n_calls", "state_mean_ms", "state_p50",
    "state_p95", "state_p99"
]
next_state_cols = [
    "next_state_n_calls", "next_state_mean_ms",
    "next_state_p50", "next_state_p95", "next_state_p99"
]
required = state_cols + next_state_cols + [
    "action", "reward", "done"
]

missing = set(required) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if df.empty or df[required].isna().any().any():
    raise ValueError("Transitions are empty or contain missing values.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with h5py.File(OUTPUT, "w") as f:
    f.attrs["description"] = (
        "Offline transition proxy from aggregated latency windows; "
        "not live on-policy experience."
    )
    f.attrs["source_csv"] = csv_path.name

    f.create_dataset(
        "observations",
        data=df[state_cols].to_numpy(dtype=np.float32)
    )
    f.create_dataset(
        "actions",
        data=df["action"].to_numpy(dtype=np.int64)
    )
    f.create_dataset(
        "rewards",
        data=df["reward"].to_numpy(dtype=np.float32)
    )
    f.create_dataset(
        "next_observations",
        data=df[next_state_cols].to_numpy(dtype=np.float32)
    )
    f.create_dataset(
        "terminals",
        data=df["done"].to_numpy(dtype=np.float32)
    )

print("Created:", OUTPUT)
print("Transitions:", len(df))
print("Observation shape:", (len(df), len(state_cols)))
print("Source CSV:", csv_path.name)
