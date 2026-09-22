"""
Step 1 of the fraud-risk model: PaySim preprocessing.

    python src/ml_model/preprocessing.py

Reads   dataset/raw/paysim/*.csv
Writes  dataset/processed/paysim_features.parquet
        (cleaned rows + engineered features + simulated time index + time-based train/val/test split)

Why each step (details in docs/data_dictionary.md):
  - account names -> integer ids          privacy: we never need who, only "same account or not"
  - hourly `step` -> `ts_sec`              the RL environment needs a clock; PaySim only has an hour counter
  - features                               inputs of the fraud-risk model and of the RL state
  - split by TIME, not randomly            the test set must be "the future", otherwise results are inflated
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make src/ importable (config, ...)
from config import OUT, PAYSIM_RAW, SEED, TRAIN_MAX_STEP, VAL_MAX_STEP  # noqa: E402

TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
FEATURES = [f"type_{t}" for t in TYPES] + ["log_amount", "orig_bal_err", "orig_zeroed", "dest_is_merchant"]
EXPECTED = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
            "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]


def preprocess(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Raw PaySim rows -> cleaned rows with features, time index and split."""
    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        raise SystemExit(f"PaySim is missing columns {missing}; found {list(df.columns)}")
    rng = np.random.default_rng(seed)

    # 1. rename to consistent snake_case (raw names mix 'Org' and 'Orig')
    df = df.rename(columns={
        "oldbalanceOrg": "old_bal_orig", "newbalanceOrig": "new_bal_orig",
        "oldbalanceDest": "old_bal_dest", "newbalanceDest": "new_bal_dest",
        "isFraud": "is_fraud", "isFlaggedFraud": "is_flagged_fraud"})

    # 2. privacy: account names -> integer surrogate keys; keep only the merchant flag
    df["dest_is_merchant"] = df["nameDest"].str.startswith("M").astype("int8")
    df["orig_id"] = pd.factorize(df["nameOrig"])[0].astype("int32")
    df["dest_id"] = pd.factorize(df["nameDest"])[0].astype("int32")
    df = df.drop(columns=["nameOrig", "nameDest"])

    # 3. time index: PaySim only has an hourly `step`; spread transactions randomly inside the hour
    df["ts_sec"] = (df["step"] - 1) * 3600 + rng.uniform(0, 3600, len(df))
    df = df.sort_values("ts_sec").reset_index(drop=True)
    df.insert(0, "txn_id", np.arange(len(df), dtype="int64"))

    # 4. feature engineering
    for t in TYPES:
        df[f"type_{t}"] = (df["type"] == t).astype("int8")
    df["log_amount"] = np.log1p(df["amount"])
    sign = np.where(df["type"] == "CASH_IN", 1.0, -1.0)        # CASH_IN credits the origin, every other type debits it
    df["orig_bal_err"] = ((df["old_bal_orig"] + sign * df["amount"] - df["new_bal_orig"]).abs() > 0.01).astype("int8")
    df["orig_zeroed"] = ((df["new_bal_orig"] == 0) & (df["old_bal_orig"] > 0)).astype("int8")

    # 5. time-based split (train = past, test = future)
    df["split"] = np.select([df["step"] <= TRAIN_MAX_STEP, df["step"] <= VAL_MAX_STEP], ["train", "val"], "test")
    return df


def design_matrix(df: pd.DataFrame, log_mean: float, log_std: float) -> pd.DataFrame:
    """Model input: the FEATURES, with log_amount standardised using TRAIN statistics only."""
    X = df[FEATURES].astype("float32").copy()
    X["log_amount"] = (X["log_amount"] - log_mean) / log_std
    return X


def main():
    files = list(PAYSIM_RAW.glob("*.csv"))
    if not files:
        raise SystemExit("No PaySim CSV found. Run: python src/simulation/download_data.py")
    df = preprocess(pd.read_csv(files[0]))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "paysim_features.parquet", index=False)
    print(f"rows: {len(df):,}   fraud rate: {df['is_fraud'].mean():.4%}")
    print(df.groupby("split").agg(rows=("txn_id", "size"), fraud=("is_fraud", "sum"), steps_min=("step", "min"), steps_max=("step", "max")))
    print("saved", OUT / "paysim_features.parquet")


if __name__ == "__main__":
    main()
