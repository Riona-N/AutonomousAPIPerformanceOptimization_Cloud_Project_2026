"""
Prepare the repo folder for a git push.

  1. creates a small dev sample of the linkage table (100k transactions per split, kept in time order)
  2. checks that nothing you are about to commit is too big for GitHub (hard limit 100 MB per file; we stay under 50 MB)
  3. checks that every file the team needs is present

    python src/simulation/prepare_for_git.py

Prints PASS / FAIL per check and exits with code 1 if anything fails.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make src/ importable (config, common, linkage_lib)
from config import OUT, ROOT  # noqa: E402

SAMPLE = OUT / "linkage_tx_sample.parquet"
LIMIT_MB = 50
MY_DIRS = ["dataset", "src", "results", "docs"]
IGNORED_PARTS = {".venv", "__pycache__", ".git", "duckdb_tmp", "paysim", "alibaba"}     # raw data folders are never committed
IGNORED_FILES = {"paysim_features.parquet", "paysim_clean.parquet", "paysim_train_undersampled.parquet", "linkage_tx.parquet", ".DS_Store"}
REQUIRED = (
    ["dataset/Dataset_Details.docx", "dataset/raw/README.md", "dataset/.gitignore",
     "src/config.py", "src/common.py", "src/linkage_lib.py", "src/.gitignore",
     "src/ml_model/preprocessing.py", "src/ml_model/train.py", "src/ml_model/predict.py", "src/ml_model/model.pkl",
     "results/accuracy.xlsx", "docs/Research_Gap_Student3.docx", "docs/README_student3.md",
     "docs/HANDOFF_ROLE1.md", "docs/data_dictionary.md"]
    + [f"src/simulation/{n}.py" for n in ("download_data", "explore", "preprocess_alibaba", "build_linkage", "validate",
                                          "tune_thresholds", "role1_quickstart", "make_graphs", "prepare_for_git")]
    + [f"dataset/processed/{n}" for n in ("stage_latency_windows.parquet", "stage_candidates.parquet", "slot_to_window.parquet",
                                          "linkage_meta.json", "alibaba_meta.json", "paysim_meta.json", "linkage_tx_sample.parquet")]
)
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def main():
    full = OUT / "linkage_tx.parquet"
    if full.exists():
        tx = pd.read_parquet(full)
        parts = [tx[tx["split"] == s].sort_values("txn_id").head(100_000) for s in ("train", "val", "test")]
        pd.concat(parts).to_parquet(SAMPLE, index=False)
        print(f"created {SAMPLE.name}: {sum(len(p) for p in parts):,} rows")
    elif not SAMPLE.exists():
        print("Neither linkage_tx.parquet nor the sample exists. Run src/simulation/build_linkage.py first.")

    files = []
    for d in MY_DIRS:
        for p in (ROOT / d).rglob("*"):
            rel = p.relative_to(ROOT)
            keep_raw_readme = rel.as_posix() == "dataset/raw/README.md"
            if p.is_file() and (keep_raw_readme or not (set(rel.parts) & IGNORED_PARTS)) and p.name not in IGNORED_FILES:
                files.append((p.stat().st_size / 1e6, rel.as_posix()))
    files.sort(reverse=True)
    print("\nlargest files that WOULD be committed:")
    for mb, name in files[:6]:
        print(f"  {mb:8.2f} MB  {name}")
    total = sum(mb for mb, _ in files)
    print()
    check(f"no file to commit is larger than {LIMIT_MB} MB", all(mb < LIMIT_MB for mb, _ in files), f"largest {files[0][0]:.1f} MB")
    check("total size to commit is under 100 MB", total < 100, f"{total:.1f} MB in {len(files)} files")
    missing = [r for r in REQUIRED if not (ROOT / r).exists()]
    check("every file the team needs is present", not missing, f"missing: {missing}" if missing else "")
    sys.exit(1 if results.count(False) else 0)


if __name__ == "__main__":
    main()
