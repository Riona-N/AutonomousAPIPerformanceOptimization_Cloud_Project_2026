
import pandas as pd


def chronological_split(
    windows,
    train_ratio=0.70,
    validation_ratio=0.15,
):
    """
    Split data by time without shuffling.

    All stages and candidates belonging to the same window_id
    stay in the same split.
    """

    if train_ratio <= 0 or validation_ratio < 0:
        raise ValueError("Invalid split ratios.")

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "Training and validation ratios must sum to less than 1."
        )

    if "window_id" not in windows.columns:
        raise ValueError("The dataset must contain window_id.")

    # Sort unique window IDs in chronological order.
    window_ids = sorted(windows["window_id"].unique())
    total_windows = len(window_ids)

    if total_windows < 3:
        raise ValueError("At least 3 unique windows are required.")

    train_end = int(total_windows * train_ratio)
    test_start = int(
        total_windows * (1 - validation_ratio)
    )

    # Keep every candidate and stage from a window together.
    train_ids = window_ids[:train_end]
    validation_ids = window_ids[train_end:test_start]
    test_ids = window_ids[test_start:]

    train = windows[
        windows["window_id"].isin(train_ids)
    ].copy()

    validation = windows[
        windows["window_id"].isin(validation_ids)
    ].copy()

    test = windows[
        windows["window_id"].isin(test_ids)
    ].copy()

    return train, validation, test


def print_split_summary(train, validation, test):
    """Display the window IDs assigned to each split."""

    splits = {
        "Training": train,
        "Validation": validation,
        "Testing": test,
    }

    for name, data in splits.items():
        ids = sorted(data["window_id"].unique())

        print(f"\n{name}")
        print(f"  Unique windows: {len(ids)}")
        print(f"  Window IDs: {ids}")
        print(f"  Total rows: {len(data)}")


if __name__ == "__main__":
    from pathlib import Path

    project_dir = Path(__file__).resolve().parent.parent
    data_file = (
        project_dir
        / "processed"
        / "stage_latency_windows.parquet"
    )

    windows = pd.read_parquet(data_file)

    train, validation, test = chronological_split(windows)

    print_split_summary(train, validation, test)

    # Verify that the splits do not share window IDs.
    train_ids = set(train["window_id"].unique())
    validation_ids = set(validation["window_id"].unique())
    test_ids = set(test["window_id"].unique())

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)

    print("\nSUCCESS: No window IDs overlap between splits.")