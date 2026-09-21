# Data Dictionary - Autonomous API Performance Optimization Framework (Role 3)

Owner: Role 3 (Data & Evaluation Engineer). Audience: Role 1 (state/action space design) and the report.
Values marked *expected* are from the dataset documentation; `02_explore.py` prints the real ones - correct this file if they differ.

---

## 1. Raw dataset A - PaySim (transaction layer)

Kaggle `ealaxi/paysim1`, CC0 licence. One CSV, *expected* 6,362,620 rows x 11 columns, ~470 MB. Synthetic mobile-money transactions over ~30 days.

| Column | Type | Meaning | Notes |
|---|---|---|---|
| `step` | int | Time unit: 1 step = 1 hour | 1..743. The ONLY time information (no timestamp) |
| `type` | str | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER | Underscores in the real file |
| `amount` | float | Transaction amount (local currency) | Heavily right-skewed |
| `nameOrig` | str | Customer who started the transaction | Anonymous id, `C...` |
| `oldbalanceOrg` | float | Origin balance before | Note spelling `Org` |
| `newbalanceOrig` | float | Origin balance after | Note spelling `Orig` |
| `nameDest` | str | Recipient | `C...` customer or `M...` merchant |
| `oldbalanceDest` | float | Recipient balance before | Not tracked for merchants (`M...`) |
| `newbalanceDest` | float | Recipient balance after | Not tracked for merchants |
| `isFraud` | 0/1 | Fraud label from the simulator | *Expected* ~0.13% positive; only in TRANSFER and CASH_OUT |
| `isFlaggedFraud` | 0/1 | Weak business rule: single transfer above 200,000 | *Expected* only ~16 rows - do NOT use as a label |

**What PaySim does NOT contain:** latency, timestamp, service/route, transaction success/failure, cost. Those must come from the linkage layer.

---

## 2. Raw dataset B - Alibaba Cluster Trace, Microservices v2021 (infrastructure layer)

12 hours of production traces, 20,000+ microservices, 10,000+ bare-metal nodes. Total ~61 GB across four tables; we download a slice. Service and node names are hashed/anonymised.

| Table (file prefix) | Grain | Columns | Used now? |
|---|---|---|---|
| `MSCallGraph` (~25 GB) | one row per call, **0.5% sampled** call graphs | `timestamp`(ms 0..43,200,000), `traceid`, `rpcid`, `um`, `rpctype`, `interface`, `dm`, `rt` | **Yes - latency source** |
| `MSRTQps` (~19 GB) | per service instance, long format, 60 s | `timestamp`, `msname`, `msinstanceid`, `metrics`, `value` | Later (true call rate, per-service RT) |
| `MSResource` (~16 GB) | per container, 60 s | `timestamp`, `msname`, `msinstanceid`, `nodeid`, `cpu_utilization`, `memory_utilization` | Later (CPU/memory state features) |
| `Node` (~1.1 GB) | per bare-metal node, 30 s | `timestamp`, `nodeid`, `cpu_utilization`, `memory_utilization` | Later (node load) |

### MSCallGraph columns

| Column | Meaning | Gotcha |
|---|---|---|
| `timestamp` | ms since trace start (0..43.2M) | One 5-minute file per `MSCallGraph_i` |
| `traceid` | One id per call graph (= one user request) | |
| `rpcid` | Position in the call tree, e.g. `0.1.1.2` | Number of dots + 1 = depth. RPC calls are logged **twice** (caller and callee) with the same `rpcid` |
| `um` / `dm` | Upstream (caller) / downstream (callee) service | Missing values appear as `NaN`, `(?)`, or empty |
| `rpctype` | `rpc`, `http`, `mq`, `db`, `mc`, ... | `db` / `mc` = data-store call |
| `interface` | Interface of the callee | Identifies the "online service" |
| `rt` | Response time in **ms** | Caller side positive, callee side **negative**; `0` means < 1 ms |

**What Alibaba does NOT contain:** an error/status column (so no real failures), payment semantics (names are hashed), exact request rates (0.5% sample).

---

## 3. Derived tables (produced by the pipeline, in `dataset/processed/`)

### `paysim_features.parquet` and `paysim_clean.parquet` - one row per transaction
`paysim_features.parquet` is written by `src/ml_model/preprocessing.py`; `paysim_clean.parquet` is the same table plus `fraud_risk`, written by `src/ml_model/predict.py`. Both are regenerated locally (too large for GitHub).
| Column | Meaning |
|---|---|
| `txn_id` | Unique, ordered by time |
| `step`, `ts_sec` | Hour (1..743) and simulated second since start (uniform random offset inside the hour) |
| `type`, `type_<TYPE>` | Original type + one-hot flags |
| `amount`, `log_amount` | Amount and log1p(amount) |
| `old_bal_orig`, `new_bal_orig`, `old_bal_dest`, `new_bal_dest` | Renamed balances |
| `orig_id`, `dest_id` | Integer surrogate keys (names removed for privacy) |
| `dest_is_merchant` | 1 if `nameDest` started with `M` |
| `orig_bal_err` | 1 if origin balance change does not equal the amount |
| `orig_zeroed` | 1 if origin account was emptied |
| `is_fraud`, `is_flagged_fraud` | Labels (unchanged) |
| `split` | `train` (steps 1-520) / `val` (521-631) / `test` (632-743) - time-ordered |
| `fraud_risk` | 0..1 score from a class-weighted logistic regression trained on `train` only |

### `stage_latency_windows.parquet` - the "live metrics" table (CloudWatch analogue)
Grain: (stage, candidate, 60 s window).
| Column | Meaning |
|---|---|
| `stage` | `ingress`, `auth`, `routing`, `processor`, `db` |
| `cand_rank` | 0 = busiest of the 3 chosen services in that stage (the static choice); 1, 2 = alternative routing targets. The 3 are chosen to be comparable in speed (caveat 6) |
| `window_id` | 60 s window of trace time |
| `n_calls` | Sampled calls in the cell (relative load, not true throughput) |
| `mean_ms`, `p50`, `p95`, `p99` | Response-time statistics |
| `q00` ... `q999` | Quantile grid (min, P5, P10, P25, P50, P75, P90, P95, P99, P99.9); the latency model samples from it |
| `imputed` | 1 if the cell had too few calls and was filled from a neighbour cell |
| `sla_violation` | 1 if window P95 > `SLA_MULT` x the stage's typical P95 |

### `stage_candidates.parquet`
`stage`, `ms` (anonymised Alibaba service id), `n_calls`, `cand_rank`, `level_p95` (long-run speed before normalisation), `coverage` (share of windows with enough calls), `scale_factor` (normalisation applied so the 3 candidates share one speed level) - which real service plays which role.

### `linkage_tx.parquet` - one row per transaction after linkage
`txn_id, step, ts_sec, type, amount, is_fraud, fraud_risk, split, slot, window_id` plus the **static-policy baseline outcome**:
`e2e_ms_static` (sum of 5 stage latencies), `sla_violation_static`, `success_static`.

### `slot_to_window.parquet`
`slot` (minute of PaySim time) -> `window_id` (Alibaba window). The RL environment steps through slots.

### JSON files
`paysim_meta.json` (splits, class counts, class weight, fraud-model threshold), `alibaba_meta.json` (data-quality counts, per-stage SLA in ms, and candidate-selection diagnostics: speed spread of the 3 busiest services vs the 3 chosen), `linkage_meta.json` (SLA/timeout in ms, baseline metrics).

---

## 4. For Role 1: what is available for state, action, reward

| Design element | Available in data | Comes from somewhere else |
|---|---|---|
| State (per stage, per candidate) | `p50`, `p95`, `p99`, `n_calls`, `sla_violation` of the current window | CPU/memory (add `MSResource`/`Node`), live AWS metrics later |
| State (per transaction) | `type`, `log_amount`, `fraud_risk`, `dest_is_merchant` | |
| Action: routing | choose `cand_rank` 0..2 at each stage; latency via `LatencyModel.sample()` | |
| Action: scaling / prioritisation | - | Effect of scaling is **not** in the data; must be modelled by the environment or measured on AWS |
| Reward: latency, SLA violation, success | `e2e` from `LatencyModel`, thresholds in `linkage_meta.json` | |
| Reward: fraud-risk term | `fraud_risk`, `is_fraud` | |

## 5. Known caveats (say these in the viva before anyone asks)

1. **Success/failure is simulated.** Neither dataset has it. A transaction "fails" when simulated latency exceeds `TIMEOUT_MULT` x the baseline median. Results are therefore relative (agent vs baselines on the same simulated backend), not absolute claims about real gateways.
2. **Alibaba is not a payment system.** Its services are hashed; the five stages are assigned by call-tree depth. We reuse its *latency behaviour*, not its business logic.
3. **Call graphs are a 0.5% sample.** `n_calls` is a relative load signal only.
4. **Time scales differ** (PaySim 30 days, Alibaba a few hours) and are aligned by load matching, not by real calendar time.
5. **PaySim fraud is easy** (fraud empties the origin account), so `fraud_risk` metrics look near-perfect. It is a simulator signal, not a fraud-detection result.
6. **Candidates are stand-ins for replicas of one service.** Per stage we take the 10 busiest Alibaba services and choose the 3 whose long-run P95 is closest, then rescale any leftover gap so all three share one speed level. Reason: the 3 busiest services differ permanently in speed, and then a one-line rule ("pick the fastest") solves routing and there is nothing left to learn. After this step candidates differ only in how they fluctuate over time. The diagnostics in `alibaba_meta.json` show the speed spread before and after.
7. **Latency model is empirical, not parametric.** A lognormal fit was tried first and rejected: real response times are a lump of near-zero calls plus a long tail, and it overestimated P95 by about 2x. Stage latencies inside one window are drawn independently, so within-request correlation between stages is not modelled.
