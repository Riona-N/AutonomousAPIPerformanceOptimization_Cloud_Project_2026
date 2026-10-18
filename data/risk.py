
"""
Reusable fraud-risk scoring interface for SENTINEL-Route.

Usage:
    from data.risk import fraud_risk
    risk = fraud_risk(txn_df)

The input must contain the features expected by the trained model.
Returns one probability-like fraud-risk score per row.
Does not modify the input or write output files.
"""

import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

# Make the existing src modules importable from this file.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
ML_PATH = SRC_PATH / "ml_model"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

if str(ML_PATH) not in sys.path:
    sys.path.insert(0, str(ML_PATH))

from ml_model.predict import load_model, score


@lru_cache(maxsize=1)
def _load_bundle():
    """Load the existing trained model once per Python process."""
    return load_model()


def fraud_risk(txn_df: pd.DataFrame):
    """
    Return the existing model's fraud-risk scores for each transaction.

    Scores are in [0, 1] and follow the input row order.
    The input must contain the model's required feature columns.
    """
    if not isinstance(txn_df, pd.DataFrame):
        raise TypeError("txn_df must be a pandas DataFrame")

    bundle = _load_bundle()
    return score(txn_df, bundle)
