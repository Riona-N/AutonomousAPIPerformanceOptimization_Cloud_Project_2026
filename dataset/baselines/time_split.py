import math

import pandas as pd


def chronological_split(
    windows,
    train_ratio=0.70,
    validation_ratio=0.15,
):
    """
    Split unique windows chronologically, without shuffling.

    All stages and candidates for a window remain in the same split.
    Ratios apply to unique window IDs, not individual rows.
    """
    if not math.isfinite(train_ratio) or not math.isfinite(validation_ratio):
        raise ValueError("Split ratios must be finite numbers.")

    if train_ratio <= 0 or validation_ratio < 0:
        raise ValueError("Training must be positive; validation cannot be negative.")

    if train_ratio + validation_ratio >= 1:
        raise ValueError("Training and validation ratios must sum to less than 1.")

    if "window_id" not in windows.columns:
        raise ValueError("The dataset must contain window_id.")

    if windows.empty:
        raise ValueError("Cannot split an empty dataset.")

    if windows["window_id"].isna().any():
        raise ValueError("window_id contains missing values.")

    window_ids = sorted(windows["window_id"].unique())
    total_windows = len(window_ids)

    if total_windows < 3:
        raise ValueError("At least 3 unique windows are required.")

    train_end = int(total_windows * train_ratio)
    validation_end = int(total_windows * (train_ratio + validation_ratio))

    # Ensure all three splits contain at least one window.
    train_end = max(1, train_end)
    validation_end = max(train_end + 1, validation_end)
    validation_end = min(validation_end, total_windows - 1)

    train_ids = window_ids[:train_end]
    validation_ids = window_ids[train_end:validation_end]
    test_ids = window_ids[validation_end:]

    train = windows[windows["window_id"].isin(train_ids)].copy()
    validation = windows[windows["window_id"].isin(validation_ids)].copy()
    test = windows[windows["window_id"].isin(test_ids)].copy()

    # Fail immediately if a split violates chronological isolation.
    train_set = set(train_ids)
    validation_set = set(validation_ids)
    test_set = set(test_ids)

    assert train_set.isdisjoint(validation_set)
    assert train_set.isdisjoint(test_set)
    assert validation_set.isdisjoint(test_set)
    assert max(train_ids) < min(validation_ids)
    assert max(validation_ids) < min(test_ids)

    return train, validation, test


def print_split_summary(train, validation, test):
    """Print split sizes and chronological window boundaries."""
    for name, data in (
        ("Training", train),
        ("Validation", validation),
        ("Testing", test),
    ):
        ids = sorted(data["window_id"].unique())
        print(f"\n{name}")
        print(f"  Unique windows: {len(ids)}")
        print(f"  Window range: {ids[0]} to {ids[-1]}" if ids else "  No windows")
        print(f"  Total rows: {len(data)}")


if __name__ == "__main__":
    from pathlib import Path

    project_dir = Path(__file__).resolve().parents[1]
    data_file = project_dir / "processed" / "stage_latency_windows.parquet"

    if not data_file.exists():
        raise SystemExit(f"Dataset not found: {data_file}")

    windows = pd.read_parquet(data_file)
    train, validation, test = chronological_split(windows)

    print_split_summary(train, validation, test)

    print("\nWindow split proportions:")
    total = windows["window_id"].nunique()
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        count = part["window_id"].nunique()
        print(f"  {name}: {count}/{total} windows ({count / total:.1%})")

    print("\nSUCCESS: splits are chronological and share no window IDs.")
