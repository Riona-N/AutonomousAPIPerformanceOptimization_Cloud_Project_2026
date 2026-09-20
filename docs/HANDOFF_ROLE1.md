# Handoff: Data & Evaluation (Role 3) -> RL & Algorithm (Role 1)

**What you are getting:** a simulated payment-gateway backend you can train and test the agent against, built from real data.
It answers one question: *"if the agent sends this transaction through service X at each stage, while the backend is in state Y, how long does it take and does it succeed?"*

```
PaySim transactions --+                       +--> your environment (state, action, reward)
 (what arrives, when) |                       |
                      +--> LINKAGE (Role 3) --+
Alibaba trace --------+                       +--> Role 3 baselines + evaluation on the SAME environment
 (how the backend behaves)
```

Start with `python src/simulation/role1_quickstart.py` (2 minutes). It loads everything and compares three toy policies.

> **Status (v2):** the 3 candidate services per stage are now chosen to be comparable in speed, so routing is a real decision (see `data_dictionary.md`, caveat 6). SLA / timeout thresholds are still provisional until we agree them on the call.

---

## 1. What is in the package

| File | What it is | Need it for |
|---|---|---|
| `data_dictionary.md` | Every field, what exists and what does NOT | Designing state / action space (read first) |
| `linkage_meta.json` | SLA and timeout thresholds in ms, baseline metrics | Reward terms |
| `alibaba_meta.json` | Per-stage SLA in ms, data-quality counts | Reward, report |
| `stage_candidates.parquet` | Which anonymised Alibaba service plays which (stage, candidate) | Reference, report |
| `stage_latency_windows.parquet` | Per (stage, candidate service, 60 s window): P50/P95/P99, call count, quantile grid | State features + latency model |
| `slot_to_window.parquet` | Minute of simulated time -> backend window | Stepping through time |
| `linkage_tx.parquet` (or `_sample`) | Every transaction: type, amount, `fraud_risk`, split, window | Traffic for the environment |
| `src/linkage_lib.py` (+ `src/config.py`) | `LatencyModel.sample(stage, candidate, windows, ...)` | The environment's "physics" |
| `src/simulation/role1_quickstart.py` | Runnable demo of the whole interface | Getting started |
| `paysim_meta.json` | Split boundaries, class counts | Reference |

`linkage_tx_sample.parquet` is a small dev sample (100k rows per split). Use it while designing, then switch to the full file.

### Getting the data

- **In the repo, enough to start:** everything under `dataset/processed/`, including `linkage_tx_sample.parquet` (100k transactions per split). `src/simulation/role1_quickstart.py` uses it automatically.
- **Full 6.3M-row table (for final training), regenerate in ~10 min, deterministic:**
  ```
  pip install pandas pyarrow duckdb scikit-learn numpy kagglehub matplotlib openpyxl
  python src/simulation/download_data.py --calls 0    # PaySim only (~470 MB). If Kaggle asks for a login, see dataset/raw/README.md
  python src/ml_model/preprocessing.py                # clean + features + time index + split
  python src/ml_model/train.py                        # fraud-risk model (model.pkl is already in the repo; this re-trains it)
  python src/ml_model/predict.py                      # adds fraud_risk -> paysim_clean.parquet
  python src/simulation/build_linkage.py              # writes dataset/processed/linkage_tx.parquet
  ```
  You do NOT need the Alibaba raw files: their processed output (`stage_latency_windows.parquet`) is already in the repo. `src/ml_model/model.pkl` (the fraud-risk model) is in the repo too, so you can skip `train.py` and only run `predict.py`.

## 2. What the data can and cannot give you

| Element | Available | Not in the data |
|---|---|---|
| State (per stage x candidate) | P50/P95/P99, call count, SLA flag of a window | CPU / memory (can be added from other Alibaba tables) |
| State (per transaction) | `type`, `amount`, `fraud_risk` | Anything about the customer beyond that |
| Action: routing | Candidate 0, 1 or 2 at each of the 5 stages (ingress, auth, routing, processor, db) | - |
| Action: scaling, prioritisation | - | **Effect of scaling is not in the data.** You (or Role 2 from AWS measurements) must model it |
| Reward: latency, SLA, success | Sampled end-to-end latency; thresholds in `linkage_meta.json` | - |
| Reward: fraud-risk term | `fraud_risk` (0..1), `is_fraud` label | - |

## 3. Rules that keep the evaluation fair

1. **Split by time.** Train on `split == "train"`, tune on `val`, report on `test`. Never touch `test` while developing.
2. **Do not change SLA or timeout thresholds locally.** Every baseline and your agent are scored against the same numbers in `linkage_meta.json`. If they should change, tell Role 3, we regenerate, everyone updates.
3. **Do not feed `is_fraud` or any `*_static` column to the agent as input.** They are labels and baseline outcomes, not observations.
4. **Observe with a lag.** The agent should see last minute's metrics (as CloudWatch would deliver them), not the current window's. `src/simulation/role1_quickstart.py` does this.
5. **Use common random numbers** when comparing policies: pass a fixed uniform matrix through `lm.sample(..., u=U[:, stage])`, as in the quickstart. Otherwise sampling luck can look like policy quality.

## 4. What I need back from you

To run the baselines and the final comparison on the same environment, I need:

- **A policy interface.** Proposal: a function `policy(obs) -> cand`, where `cand` is one candidate index per stage, chosen at the granularity you decide. Baselines (static, rule-based, bandit) use the same signature.
- **The exact observation vector** you feed the agent, so the baselines see the same information.
- **Your trained model file plus a 5-line "how to load and call it"** for the evaluation script.

## 5. Known limitations (details in `data_dictionary.md`, section 5)

- Success/failure is **simulated** (timeout rule). Neither dataset has real failures. Results are relative, not absolute.
- Alibaba services are anonymised; the 5 stages are assigned by call-tree depth, and the 3 candidates per stage are chosen (from the 10 busiest services) to be comparable in speed and any leftover gap is normalised, so they are stand-ins for replicas of one service and differ only in how they fluctuate over time.
- Latencies of the 5 stages inside one window are drawn independently.
- `fraud_risk` looks near-perfect because PaySim fraud is easy (fraud empties the account). Treat it as a simulator signal.

## 6. To settle in one 15-minute call

1. Thresholds: defaults are `SLA_MULT = 2.0` and `TIMEOUT_MULT = 4.0`; final values will be set together after the environment is regenerated.
2. Decision granularity: one decision per transaction, or one per minute applied to all traffic?
3. Action space: 3 candidates x 5 stages = 243 combinations. Do you want to reduce it (e.g. one choice applied to all stages)?
4. Who models the effect of scaling and prioritisation (you, or Role 2 with AWS measurements)?
5. Reward weights and how the fraud-risk term enters.

*Status: thresholds and the choice of trace slice are provisional. If they change, Role 3 regenerates and pushes; pull before you train.*
