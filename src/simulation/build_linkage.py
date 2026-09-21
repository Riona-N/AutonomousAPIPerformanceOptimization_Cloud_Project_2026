"""
Step 5 - the synthetic LINKAGE LAYER: join PaySim transactions to Alibaba backend behaviour.

PaySim says WHAT traffic arrives and WHEN.  Alibaba says HOW SLOW the backend is at a given moment.
Neither dataset contains the other, so we connect them with two explicit, documented rules:

  Rule A - load matching (time alignment).
      PaySim spans 30 days, Alibaba only a few minutes-hours. We rank PaySim hours by traffic volume and
      Alibaba windows by how slow they are, then map busy hours -> slow windows and quiet hours -> calm
      windows (with a little random jitter). Rush hour therefore hits a stressed backend, which is exactly
      the situation the RL agent has to learn to handle.
  Rule B - path + latency.
      Every transaction traverses the 5-stage chain. Its latency at each stage is drawn from the real
      Alibaba response-time distribution of the chosen service in the matched window. Here we apply the
      STATIC policy (always candidate 0) to get the baseline outcome for every transaction.
      Latency > SLA_MULT x median -> SLA violation; latency > TIMEOUT_MULT x median -> transaction fails.

Output (dataset/processed/):
  linkage_tx.parquet        one row per transaction: window, static-policy latency / SLA flag / success flag
  slot_to_window.parquet    minute-slot -> Alibaba window (what the RL environment steps through)
  linkage_meta.json         SLA / timeout thresholds in ms and headline baseline numbers
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make src/ importable (config, common, linkage_lib)
import json

import numpy as np
import pandas as pd

from config import OUT, SEED, SLA_MULT, STAGES, TIMEOUT_MULT
from linkage_lib import LatencyModel


def main():
    rng = np.random.default_rng(SEED)
    tx = pd.read_parquet(OUT / "paysim_clean.parquet",
                         columns=["txn_id", "step", "ts_sec", "type", "amount", "is_fraud", "fraud_risk", "split"])
    lm = LatencyModel(OUT / "stage_latency_windows.parquet")
    n_win = len(lm.windows)

    # ---- Rule A: load matching at hour level, jitter at minute level ---------------------------------
    hourly = tx.groupby("step").size().reindex(range(1, int(tx["step"].max()) + 1))
    rank = hourly.rank(pct=True).fillna(0.5)                       # 0..1 (busy hours near 1)
    hour_pos = (np.ceil(rank * n_win).astype(int) - 1).clip(0, n_win - 1)   # position in calm->busy ordering
    order = np.argsort(lm.stress_by_window())                     # window positions sorted calm -> busy

    n_slots = int(tx["ts_sec"].max() // 60) + 1
    slot_hour = np.arange(n_slots) // 60 + 1
    base = hour_pos.reindex(slot_hour).fillna(n_win // 2).to_numpy().astype(int)
    jitter = max(1, n_win // 10)
    slot_pos = np.clip(base + rng.integers(-jitter, jitter + 1, n_slots), 0, n_win - 1)
    slot_window = lm.windows[order[slot_pos]]
    slot_to_window = pd.DataFrame({"slot": np.arange(n_slots), "window_id": slot_window})
    slot_to_window.to_parquet(OUT / "slot_to_window.parquet", index=False)

    tx["slot"] = (tx["ts_sec"] // 60).astype(int)
    tx["window_id"] = slot_window[tx["slot"].to_numpy()]

    # ---- Rule B: static-policy latency through the 5-stage chain ----------------------------------------
    lat = np.zeros((len(tx), len(STAGES)), dtype=np.float32)
    for s in range(len(STAGES)):
        lat[:, s] = lm.sample(s, 0, tx["window_id"].to_numpy(), rng)
    e2e = lat.sum(axis=1)
    base_median = float(np.median(e2e))
    sla_ms, timeout_ms = SLA_MULT * base_median, TIMEOUT_MULT * base_median

    tx["e2e_ms_static"] = e2e
    tx["sla_violation_static"] = (e2e > sla_ms).astype("int8")
    tx["success_static"] = (e2e <= timeout_ms).astype("int8")
    tx.to_parquet(OUT / "linkage_tx.parquet", index=False)

    meta = {
        "baseline_median_e2e_ms": round(base_median, 2), "sla_ms": round(sla_ms, 2), "timeout_ms": round(timeout_ms, 2),
        "static_policy": {
            "sla_violation_rate": round(float(tx["sla_violation_static"].mean()), 4),
            "success_rate": round(float(tx["success_static"].mean()), 4),
            "p95_e2e_ms": round(float(np.percentile(e2e, 95)), 2),
            "p99_e2e_ms": round(float(np.percentile(e2e, 99)), 2)},
        "by_split_sla_violation": tx.groupby("split")["sla_violation_static"].mean().round(4).to_dict(),
        "n_alibaba_windows": n_win, "n_minute_slots": n_slots,
    }
    (OUT / "linkage_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
