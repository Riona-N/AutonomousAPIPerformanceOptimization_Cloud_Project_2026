"""
Step 3 of the fraud-risk model: score transactions with the trained model.

    python src/ml_model/predict.py

Reads   dataset/processed/paysim_features.parquet  and  src/ml_model/model.pkl
Writes  dataset/processed/paysim_clean.parquet     (same rows + a `fraud_risk` column between 0 and 1)

In your own code:
    from predict import load_model, score
    bundle = load_model()
    risk = score(df_with_features, bundle)          # numpy array, one value per row

`paysim_clean.parquet` is the input of the linkage layer (src/simulation/build_linkage.py).
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make src/ importable (config, ...)
from config import MODEL_PATH, OUT  # noqa: E402
from preprocessing import design_matrix  # noqa: E402


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def score(df: pd.DataFrame, bundle) -> "pd.Series":
    """fraud_risk (0..1) for every row of a dataframe that already has the model features."""
    X = design_matrix(df, bundle["log_amount_mean"], bundle["log_amount_std"])
    return bundle["model"].predict_proba(X)[:, 1].astype("float32")


def main():
    if not MODEL_PATH.exists():
        raise SystemExit("model.pkl not found. Run: python src/ml_model/train.py")
    bundle = load_model()
    df = pd.read_parquet(OUT / "paysim_features.parquet")
    df["fraud_risk"] = score(df, bundle)
    for s in ("val", "test"):
        m = df["split"] == s
        print(f"fraud_risk PR-AUC on {s:4s}: {average_precision_score(df.loc[m, 'is_fraud'], df.loc[m, 'fraud_risk']):.3f}")
    for c in df.select_dtypes("float64").columns.difference(["ts_sec"]):
        df[c] = df[c].astype("float32")
    df.to_parquet(OUT / "paysim_clean.parquet", index=False)
    print(f"saved {OUT / 'paysim_clean.parquet'}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
