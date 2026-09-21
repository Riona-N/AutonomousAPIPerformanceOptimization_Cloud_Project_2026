"""
Step 6 - data validation. Prints PASS / FAIL per check; exits with code 1 if anything fails.

  python src/simulation/validate.py
Paste the output into the report's data-validation section.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make src/ importable (config, common, linkage_lib)

import numpy as np
import pandas as pd

from config import OUT, STAGES, TRAIN_MAX_STEP, VAL_MAX_STEP
from linkage_lib import QCOLS, LatencyModel

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def main():
    print("--- PaySim ---")
    px = pd.read_parquet(OUT / "paysim_clean.parquet")
    check("no missing values in core columns",
          px[["txn_id", "step", "ts_sec", "type", "amount", "is_fraud", "fraud_risk", "split"]].notna().all().all())
    check("amounts are non-negative", (px["amount"] >= 0).all())
    rate = px["is_fraud"].mean()
    check("fraud rate is the expected ~0.13% (real distribution preserved)", 0.0010 <= rate <= 0.0016, f"{rate:.4%}")
    check("fraud_risk is a probability in [0, 1]", px["fraud_risk"].between(0, 1).all())
    check("txn_id unique and time-ordered", px["txn_id"].is_unique and px["ts_sec"].is_monotonic_increasing)
    tr, va, te = (px.loc[px["split"] == s, "step"] for s in ("train", "val", "test"))
    check("splits are time-ordered with no overlap", tr.max() <= TRAIN_MAX_STEP < va.min() and va.max() <= VAL_MAX_STEP < te.min(),
          f"train<= {tr.max()}, val {va.min()}-{va.max()}, test>= {te.min()}")
    fraud_types = set(px.loc[px["is_fraud"] == 1, "type"].unique())
    check("fraud only occurs in TRANSFER / CASH_OUT", fraud_types <= {"TRANSFER", "CASH_OUT"}, str(sorted(fraud_types)))
    for s in ("train", "val", "test"):
        m = px["split"] == s
        check(f"{s} split contains fraud cases", px.loc[m, "is_fraud"].sum() > 0, f"{int(px.loc[m, 'is_fraud'].sum())} cases")

    print("\n--- Alibaba backend ---")
    sw = pd.read_parquet(OUT / "stage_latency_windows.parquet")
    cand = pd.read_parquet(OUT / "stage_candidates.parquet")
    check("all 5 chain stages present", set(sw["stage"]) == set(STAGES))
    check("every stage has >= 2 routable candidates (else routing is meaningless)",
          (cand.groupby("stage").size() >= 2).all(), str(cand.groupby("stage").size().to_dict()))
    check("latency statistics have no NaN or negative values",
          sw[QCOLS].notna().all().all() and (sw[QCOLS] >= 0).all().all())
    check("quantile grid is non-decreasing everywhere (P50 <= P95 <= P99 ...)", (np.diff(sw[QCOLS].to_numpy(), axis=1) >= -1e-6).all())
    check("windows are contiguous", np.all(np.diff(np.sort(sw["window_id"].unique())) == 1),
          f"{sw['window_id'].nunique()} windows")
    lvl = sw.groupby(["stage", "cand_rank"])["p95"].median().unstack()
    ratio = lvl.max(axis=1) / lvl.min(axis=1).clip(lower=1.0)
    check("candidates in each stage are comparable in speed (max/min long-run P95 <= 1.25)", (ratio <= 1.25).all(),
          str(ratio.round(2).to_dict()))
    piv = sw.pivot_table(index="window_id", columns=["stage", "cand_rank"], values="p95")
    static_p95 = sum(piv[(s, 0)] for s in STAGES).mean()
    oracle_p95 = sum(piv[s].min(axis=1) for s in STAGES).mean()
    check("routing headroom is realistic: picking the best candidate every window gives 50%-100% of the static path P95",
          0.5 <= oracle_p95 / static_p95 <= 1.0, f"{oracle_p95 / static_p95:.0%}")
    imputed = sw["imputed"].mean()
    check("under 30% of (stage, candidate, window) cells imputed", imputed < 0.30, f"{imputed:.1%}")
    print("\n--- Linkage ---")
    lk = pd.read_parquet(OUT / "linkage_tx.parquet")
    slots = pd.read_parquet(OUT / "slot_to_window.parquet")
    check("row count matches PaySim", len(lk) == len(px), f"{len(lk):,}")
    check("every transaction maps to a real Alibaba window", lk["window_id"].isin(sw["window_id"].unique()).all())
    check("every minute slot maps to a real Alibaba window", slots["window_id"].isin(sw["window_id"].unique()).all())
    check("static-policy latencies are finite and positive", np.isfinite(lk["e2e_ms_static"]).all() and (lk["e2e_ms_static"] > 0).all())
    viol, succ = lk["sla_violation_static"].mean(), lk["success_static"].mean()
    check("SLA-violation rate is non-degenerate (0.5% - 40%): agent has something to improve", 0.005 <= viol <= 0.40, f"{viol:.2%}")
    check("success rate leaves headroom (90% - 99.9%); if FAIL, tune TIMEOUT_MULT in config.py", 0.90 <= succ <= 0.999, f"{succ:.3%}")
    busy = lk.groupby("step").size()
    hi = lk[lk["step"].isin(busy.nlargest(50).index)]["e2e_ms_static"].mean()
    lo = lk[lk["step"].isin(busy.nsmallest(50).index)]["e2e_ms_static"].mean()
    check("busy hours see slower backend than quiet hours (load matching works)", hi > lo, f"busy {hi:.0f} ms vs quiet {lo:.0f} ms")

    lm = LatencyModel(OUT / "stage_latency_windows.parquet")
    rng = np.random.default_rng(0)
    mid = lm.windows[len(lm.windows) // 2]
    for s, name in enumerate(STAGES):
        ref = sw[(sw["stage"] == name) & (sw["cand_rank"] == 0) & (sw["window_id"] == mid)].iloc[0]
        # (a) exact test: pushing u = 0.50 / 0.95 / 0.99 through the sampler must return the stored quantiles
        got = lm.sample(s, 0, np.repeat(mid, 3), u=[0.50, 0.95, 0.99])
        check(f"sampler is exact at P50/P95/P99 ({name})", np.allclose(got, [ref["q50"], ref["q95"], ref["q99"]], rtol=1e-4, atol=1e-3))
        # (b) statistical test on random draws. P50 and P95 only: sampled P99 is dominated by noise
        #     whenever P99.9 is far above P99 (heavy tail), so P99 is covered by the exact test above.
        draws = lm.sample(s, 0, np.repeat(mid, 200_000), rng)
        worst = max(abs(np.percentile(draws, q) - ref[c]) / max(ref[c], 2.0) for q, c in ((50, "p50"), (95, "p95")))
        check(f"random draws reproduce real P50/P95 ({name}, mid window)", worst < 0.15,
              f"worst error {worst:.1%}; table P95 {ref['p95']:.0f} ms, P99 {ref['p99']:.0f} ms, P99.9 {ref['q999']:.0f} ms")

    n_fail = results.count(False)
    print(f"\n{results.count(True)} passed, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
