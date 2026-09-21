"""
Step 4 - preprocess the Alibaba microservice call graph into a small "payment-gateway-like" backend.

What this does, in order:
  1. de-duplicate calls   (an RPC is logged twice: caller side rt > 0, callee side rt < 0)
  2. drop calls with a missing/anonymised downstream service name or an absurd response time
  3. give every call a STAGE of the chain  ingress -> auth -> routing -> processor -> db
     (by call depth in the call tree; data-store calls become the 'db' stage)
  4. per stage take the 10 busiest services (the pool) and pick the 3 whose long-run speed is CLOSEST.
     Routing only means something if the candidates are comparable replicas: if one were permanently 10x
     faster, "always pick it" would solve the problem and there would be nothing for an agent to learn.
     Any leftover speed gap is normalised away (config.EQUALIZE_CANDIDATES), so what differs between
     candidates is how they fluctuate over time, not how fast they are on average.
  5. per (stage, candidate, 60 s window) store a quantile grid (min, P5 ... P99.9) + P50/P95/P99 + SLA flag

Output (dataset/processed/):
  stage_latency_windows.parquet   the "live metrics" table (state features + latency model for the simulator)
  stage_candidates.parquet        which anonymised Alibaba service plays which (stage, candidate rank), with its level
  alibaba_meta.json               data-quality counts, per-stage SLA thresholds, candidate-selection diagnostics
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make src/ importable (config, common, linkage_lib)
import json
from itertools import combinations

import duckdb
import numpy as np
import pandas as pd

from common import BAD_NAMES, open_callgraph
from config import (CANDIDATE_POOL, CANDIDATES_PER_STAGE, EQUALIZE_CANDIDATES, MAX_RT_MS, MIN_CALLS_PER_CELL,
                    MIN_COVERAGE, OUT, SLA_MULT, STAGES, WINDOW_MS)
from linkage_lib import QCOLS, QUANTILE_LEVELS


def spread(levels):
    """max/min ratio of a set of speed levels (1.0 = identical)."""
    lv = np.clip(np.asarray(levels, dtype=float), 1.0, None)
    return float(lv.max() / lv.min())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "duckdb_tmp"
    tmp.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA temp_directory='{tmp.as_posix()}'")   # lets DuckDB spill to disk on small-RAM laptops
    open_callgraph(con)
    one = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731

    n_raw = one("SELECT COUNT(*) FROM cg")

    # 1. de-duplicate: one row per (traceid, rpcid). Prefer the caller-side (positive) rt, else |callee-side|.
    con.execute(f"""
        CREATE TABLE calls AS
        SELECT traceid, rpcid, MIN(ts) AS ts, MAX(rt) AS rt_max, MIN(rt) AS rt_min,
               ANY_VALUE(dm)      FILTER (WHERE dm IS NOT NULL AND dm NOT IN {BAD_NAMES}) AS dm,
               ANY_VALUE(rpctype) FILTER (WHERE rpctype IS NOT NULL)                       AS rpctype
        FROM cg
        WHERE traceid IS NOT NULL AND rpcid IS NOT NULL AND ts IS NOT NULL AND rt IS NOT NULL
        GROUP BY traceid, rpcid""")
    n_calls = one("SELECT COUNT(*) FROM calls")
    n_named = one("SELECT COUNT(*) FROM calls WHERE dm IS NOT NULL")

    # 2 + 3. clean + assign stage
    con.execute(f"""
        CREATE TABLE stage_calls AS
        SELECT CAST(FLOOR(ts / {WINDOW_MS}) AS INTEGER) AS window_id, dm AS ms, rt_ms,
               CASE WHEN rpctype IN ('db', 'mc') THEN 'db'
                    WHEN depth <= 2 THEN 'ingress'
                    WHEN depth = 3  THEN 'auth'
                    WHEN depth = 4  THEN 'routing'
                    ELSE 'processor' END AS stage
        FROM (
            SELECT ts, dm, rpctype,
                   LENGTH(rpcid) - LENGTH(REPLACE(rpcid, '.', '')) + 1 AS depth,
                   CASE WHEN rt_max > 0 THEN rt_max WHEN rt_min < 0 THEN -rt_min ELSE 0 END AS rt_ms
            FROM calls WHERE dm IS NOT NULL)
        WHERE rt_ms <= {MAX_RT_MS}""")
    n_clean = one("SELECT COUNT(*) FROM stage_calls")

    # 4a. pool = the busiest services per stage
    pool = con.execute(f"""
        SELECT * FROM (
            SELECT stage, ms, COUNT(*) AS n_calls,
                   ROW_NUMBER() OVER (PARTITION BY stage ORDER BY COUNT(*) DESC) AS vol_rank
            FROM stage_calls GROUP BY stage, ms)
        WHERE vol_rank <= {CANDIDATE_POOL}""").df()
    con.register("pool_df", pool)
    got = set(pool["stage"])
    if got != set(STAGES):
        raise SystemExit(f"Stages with no data: {set(STAGES) - got}. Download more call-graph files (--calls 12).")

    # 4b. per-window statistics for every pool service: a quantile grid per cell
    qsql = ", ".join(("MIN(s.rt_ms)" if lv == 0.0 else f"quantile_cont(s.rt_ms, {lv})") + f" AS {c}"
                     for lv, c in zip(QUANTILE_LEVELS, QCOLS))
    wp = con.execute(f"""
        SELECT c.stage, c.ms, s.window_id, COUNT(*) AS n_calls, AVG(s.rt_ms) AS mean_ms, {qsql}
        FROM stage_calls s JOIN pool_df c ON s.stage = c.stage AND s.ms = c.ms
        GROUP BY 1, 2, 3""").df()
    windows = np.arange(wp["window_id"].min(), wp["window_id"].max() + 1)

    # 4c. each service's long-run speed level (median over windows of its P95) and how many windows it covers
    solid = wp[wp["n_calls"] >= MIN_CALLS_PER_CELL]
    lvl = solid.groupby(["stage", "ms"]).agg(level=("q95", "median"), covered=("window_id", "nunique")).reset_index()
    lvl["coverage"] = lvl["covered"] / len(windows)
    pool = pool.merge(lvl[["stage", "ms", "level", "coverage"]], on=["stage", "ms"], how="left")

    # 4d. per stage: choose the 3 eligible services with the smallest max/min speed ratio
    k = CANDIDATES_PER_STAGE
    chosen, diag = [], {}
    for stage in STAGES:
        p = pool[(pool["stage"] == stage) & pool["level"].notna()].reset_index(drop=True)
        elig = p[p["coverage"] >= MIN_COVERAGE].reset_index(drop=True)
        if len(elig) < k:                                   # not enough well-covered services: use what we have
            print(f"WARNING: stage {stage} has only {len(elig)} well-covered services; using all {len(p)} available")
            elig = p
        by_volume = p.nsmallest(k, "vol_rank")
        lv, vol = np.log(elig["level"].clip(lower=1.0).to_numpy()), elig["n_calls"].to_numpy()
        if len(elig) <= k:
            best = list(range(len(elig)))
        else:
            best = list(min(combinations(range(len(elig)), k),
                            key=lambda c: (lv[list(c)].max() - lv[list(c)].min(), -vol[list(c)].sum())))
        sel = elig.iloc[best].sort_values("n_calls", ascending=False).reset_index(drop=True)
        sel["cand_rank"] = np.arange(len(sel))
        target = float(sel["level"].median())
        sel["scale_factor"] = (target / sel["level"].clip(lower=1.0)) if EQUALIZE_CANDIDATES else 1.0
        chosen.append(sel)
        diag[stage] = {"speed_spread_top3_by_volume": round(spread(by_volume["level"]), 2),
                       "speed_spread_chosen": round(spread(sel["level"]), 2),
                       "residual_rescale_applied": bool(EQUALIZE_CANDIDATES),
                       "max_scale_factor": round(float(sel["scale_factor"].max()), 2),
                       "pool_levels_ms": [round(float(x), 1) for x in sorted(p["level"])]}
    cand = pd.concat(chosen, ignore_index=True)[["stage", "ms", "n_calls", "cand_rank", "level", "coverage", "scale_factor"]]
    cand = cand.rename(columns={"level": "level_p95"})

    # 5. build the (stage, candidate, window) table for the chosen services, applying the normalisation
    win = wp.merge(cand[["stage", "ms", "cand_rank", "scale_factor"]], on=["stage", "ms"], how="inner")
    stat_cols = ["mean_ms"] + QCOLS
    win[stat_cols] = win[stat_cols].mul(win["scale_factor"], axis=0)
    win = win.drop(columns=["ms", "scale_factor"])

    # Complete grid: every (stage, candidate) x every window. Thin/missing cells are IMPUTED, not trusted.
    grid = pd.MultiIndex.from_tuples(
        [(r.stage, int(r.cand_rank), int(w)) for r in cand.itertuples() for w in windows],
        names=["stage", "cand_rank", "window_id"])
    win = win.set_index(["stage", "cand_rank", "window_id"]).reindex(grid).reset_index()
    win["n_calls"] = win["n_calls"].fillna(0).astype(int)
    win["imputed"] = (win["n_calls"] < MIN_CALLS_PER_CELL).astype("int8")
    win.loc[win["imputed"] == 1, stat_cols] = np.nan
    for cols in (["stage", "cand_rank"], ["stage"]):          # fall back to that candidate's median, then the stage's
        win[stat_cols] = win[stat_cols].fillna(win.groupby(cols)[stat_cols].transform("median"))
    win[QCOLS] = np.maximum.accumulate(win[QCOLS].to_numpy(), axis=1)   # quantiles must never decrease
    win["p50"], win["p95"], win["p99"] = win["q50"], win["q95"], win["q99"]   # convenient aliases

    # SLA flag per stage: window P95 above SLA_MULT x the stage's typical P95
    stage_sla = (win.groupby("stage")["p95"].median() * SLA_MULT).round(2)
    win["sla_violation"] = (win["p95"] > win["stage"].map(stage_sla)).astype("int8")

    win.to_parquet(OUT / "stage_latency_windows.parquet", index=False)
    cand.to_parquet(OUT / "stage_candidates.parquet", index=False)
    meta = {
        "raw_rows": int(n_raw), "unique_calls_after_dedup": int(n_calls),
        "calls_with_valid_dm": int(n_named), "pct_calls_dropped_missing_dm": round(100 * (1 - n_named / max(n_calls, 1)), 2),
        "calls_after_rt_filter": int(n_clean), "n_windows": int(len(windows)), "window_seconds": WINDOW_MS // 1000,
        "stage_sla_ms": stage_sla.to_dict(), "pct_cells_imputed": round(100 * win["imputed"].mean(), 2),
        "candidate_selection": diag,
    }
    (OUT / "alibaba_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k_: v for k_, v in meta.items() if k_ != "candidate_selection"}, indent=2))
    print("\ncandidate speed spread per stage (max/min of long-run P95; 1.0 = identical):")
    print(f"  {'stage':10s} {'top-3 by volume':>16s} {'chosen':>8s}")
    for s, d in diag.items():
        print(f"  {s:10s} {d['speed_spread_top3_by_volume']:16.2f} {d['speed_spread_chosen']:8.2f}")
    print(cand.sort_values(["stage", "cand_rank"]).assign(ms=lambda d: d["ms"].str[:10] + "...").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
