"""
Role 1 quickstart - shows how to use the data as a simulated backend for an RL agent.

    python src/simulation/role1_quickstart.py

This is a DEMO of the interface, not the environment you have to build. State, action, reward and the
decision granularity (per transaction or per minute) are YOURS to design. What it demonstrates:

  1. how to load the tables and the SLA / timeout thresholds
  2. what "the agent observes": last minute's latency stats per (stage, candidate)  (CloudWatch-style lag)
  3. what "the agent does": pick a candidate service (0, 1 or 2) for each of the 5 stages
  4. what "the environment returns": end-to-end latency, SLA violation, success for every transaction
  5. how to compare policies fairly (common random numbers), on the TEST split only

Import it from other scripts:  from role1_quickstart import evaluate_policies
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make src/ importable (config, common, linkage_lib)
from config import OUT, STAGES  # noqa: E402
from linkage_lib import QCOLS, LatencyModel  # noqa: E402

S = len(STAGES)
Q50, Q95 = QCOLS.index("q50"), QCOLS.index("q95")
DAY = 24 * 60          # one simulated day = 1440 one-minute slots


def evaluate_policies(policies=None, seed=0):
    """Play one simulated TEST day with each policy. Returns a dataframe: mean latency, P95, SLA violations, success."""
    meta = json.loads((OUT / "linkage_meta.json").read_text())
    sla_ms, timeout_ms = meta["sla_ms"], meta["timeout_ms"]           # agreed thresholds - do not change locally
    tx_file = OUT / "linkage_tx.parquet"
    if not tx_file.exists():
        tx_file = OUT / "linkage_tx_sample.parquet"                    # the small dev sample committed to the repo
    tx = pd.read_parquet(tx_file, columns=["txn_id", "slot", "window_id", "type", "amount", "fraud_risk", "is_fraud", "split"])
    slot_to_window = pd.read_parquet(OUT / "slot_to_window.parquet")["window_id"].to_numpy()
    lm = LatencyModel(OUT / "stage_latency_windows.parquet")

    # one simulated day of the TEST split (test = the "future"; never train on it)
    test = tx[tx["split"] == "test"].sort_values("slot")
    first = int(test["slot"].min())
    day = test[(test["slot"] >= first) & (test["slot"] < first + DAY)].reset_index(drop=True)
    n = len(day)
    # Common random numbers: one fixed uniform draw per (transaction, stage), reused by EVERY policy.
    # Differences between policies then come from their decisions, not from sampling luck.
    U = np.random.default_rng(seed).random((n, S))
    bounds = np.searchsorted(day["slot"].to_numpy(), np.arange(first, first + DAY + 1))
    windows = day["window_id"].to_numpy()

    def observe(slot):
        """What the agent sees at the start of a minute: LAST minute's P50 and P95 per (stage, candidate)."""
        w = lm.pos([slot_to_window[slot - 1]])[0]
        return lm.q[:, :, w, Q50], lm.q[:, :, w, Q95]                  # two arrays, shape (stages, candidates)

    def run(policy):
        """policy(p50, p95) -> array of S candidate indices (one per stage), applied to that whole minute."""
        e2e = np.zeros(n, dtype=np.float64)
        for k in range(DAY):
            lo, hi = bounds[k], bounds[k + 1]
            if lo == hi:
                continue
            cand = policy(*observe(first + k))                         # <-- the agent's decision
            for s in range(S):                                         # <-- the environment's response
                e2e[lo:hi] += lm.sample(s, int(cand[s]), windows[lo:hi], u=U[lo:hi, s])
        return e2e

    if policies is None:
        rng = np.random.default_rng(seed + 1)
        policies = {
            "static (always candidate 0)": lambda p50, p95: np.zeros(S, dtype=int),
            "random": lambda p50, p95: np.array([rng.integers(lm.n_cand[s]) for s in range(S)]),
            "greedy (lowest P95 last minute)": lambda p50, p95: np.nanargmin(p95, axis=1),
        }
    rows = []
    for name, pol in policies.items():
        e2e = run(pol)
        rows.append({"policy": name, "mean_ms": e2e.mean(), "p95_ms": np.percentile(e2e, 95),
                     "sla_violation": np.mean(e2e > sla_ms), "success": np.mean(e2e <= timeout_ms), "n_tx": n})
    res = pd.DataFrame(rows)
    res.attrs.update(sla_ms=sla_ms, timeout_ms=timeout_ms, source=tx_file.name)
    return res


def main():
    res = evaluate_policies()
    print(f"loaded transactions from {res.attrs['source']}; SLA = {res.attrs['sla_ms']:.0f} ms, timeout = {res.attrs['timeout_ms']:.0f} ms")
    print(f"\nTEST day, {int(res['n_tx'][0]):,} transactions")
    print(f"{'policy':34s} {'mean ms':>9s} {'P95 ms':>9s} {'SLA viol':>9s} {'success':>9s}")
    for r in res.itertuples():
        print(f"{r.policy:34s} {r.mean_ms:9.0f} {r.p95_ms:9.0f} {r.sla_violation:9.2%} {r.success:9.2%}")
    print("\nAlso available per transaction for your state / reward: type, amount, fraud_risk (a 0..1 signal).")
    print("Do NOT feed is_fraud or any *_static column to the agent as input: they are labels / baseline outcomes.")


if __name__ == "__main__":
    main()
