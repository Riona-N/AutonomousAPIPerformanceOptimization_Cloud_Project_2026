# Student 3 / Role 3 - Data & Evaluation

Builds the simulated payment-gateway backend that the RL agent trains and is evaluated on: PaySim transactions
(what arrives, when) linked to the Alibaba microservice trace (how the backend behaves). Also trains the
fraud-risk model whose output feeds the agent's state and reward.

**Role 1: start with `docs/HANDOFF_ROLE1.md`, then `docs/data_dictionary.md`.**

## Layout

```
dataset/
  raw/                      download instructions only (raw data is not committed)
  processed/                small outputs (committed); large tables are regenerated
  Dataset_Details.docx      dataset description for the report
src/
  config.py common.py linkage_lib.py         shared settings + the LatencyModel interface
  ml_model/                 fraud-risk model: preprocessing.py, train.py, predict.py, model.pkl
  simulation/               backend simulation: download, explore, Alibaba preprocessing, linkage, validation, graphs
results/
  graphs/  screenshots/  accuracy.xlsx
docs/
  Research_Gap_Student3.docx  README_student3.md  HANDOFF_ROLE1.md  data_dictionary.md
```

## Run order

```
pip install pandas pyarrow duckdb scikit-learn numpy kagglehub matplotlib openpyxl
```

| Step | Command (from the repo root) | What you get |
|---|---|---|
| 1 | `python src/simulation/download_data.py` (`--calls 12` for more trace, `--calls 0` for PaySim only) | Raw data in `dataset/raw/` (~1 GB) |
| 2 | `python src/simulation/explore.py` | Printed profile of both datasets |
| 3 | `python src/ml_model/preprocessing.py` | `paysim_features.parquet` |
| 4 | `python src/ml_model/train.py` | `model.pkl`, `results/accuracy.xlsx`, model graphs, `paysim_meta.json` |
| 5 | `python src/ml_model/predict.py` | `paysim_clean.parquet` (adds `fraud_risk`) |
| 6 | `python src/simulation/preprocess_alibaba.py` | `stage_latency_windows.parquet`, `stage_candidates.parquet`, `alibaba_meta.json` |
| 7 | `python src/simulation/build_linkage.py` | `linkage_tx.parquet`, `slot_to_window.parquet`, `linkage_meta.json` |
| 8 | `python src/simulation/validate.py` | PASS/FAIL data-validation report (take a screenshot for `results/screenshots/`) |
| 9 | `python src/simulation/make_graphs.py` | Simulation graphs in `results/graphs/` |
| 10 | `python src/simulation/prepare_for_git.py` | Dev sample + size checks before pushing |

Optional: `python src/simulation/tune_thresholds.py` shows what SLA / failure rates each threshold setting would give (read-only).
All settings (splits, SLA / timeout multipliers, candidate selection) live in `src/config.py`.

## If something breaks

- **"Could not find columns ..."** (steps 2 / 6): the CSV header differs from the README. Run step 2, read the `columns found` line, add the spelling to `ALIASES` in `src/common.py`.
- **"Stages with no data"** (step 6): not enough trace loaded. Run `python src/simulation/download_data.py --calls 12`.
- **Out of memory in step 6:** close other apps (DuckDB spills to `dataset/processed/duckdb_tmp`) or use fewer files.
- **Kaggle asks for credentials:** download the CSV from https://www.kaggle.com/datasets/ealaxi/paysim1 and unzip it into `dataset/raw/paysim/`.
- **The success-rate check FAILs in step 8:** not a bug. The baseline almost never fails (or fails too often). Run the tuning script, change `TIMEOUT_MULT` in `src/config.py` with Role 1, and re-run steps 7-8.
