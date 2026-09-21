"""
Step 2 - explore both datasets BEFORE preprocessing (Phase 0 deliverable: the data dictionary).

  python src/simulation/explore.py            # both
  python src/simulation/explore.py paysim
  python src/simulation/explore.py alibaba

Read the printed output and check it against data_dictionary.md. Anything that differs
(column names, value ranges, fraud rate) goes into the dictionary and into the report.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make src/ importable (config, common, linkage_lib)

import duckdb
import pandas as pd

from common import BAD_NAMES, callgraph_files, open_callgraph
from config import ALI_RAW, PAYSIM_RAW

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)


def banner(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def explore_paysim():
    banner("PAYSIM")
    files = list(PAYSIM_RAW.glob("*.csv"))
    if not files:
        print("No PaySim CSV found. Run src/simulation/download_data.py first.")
        return
    df = pd.read_csv(files[0])
    print("file  :", files[0].name)
    print("shape :", df.shape)
    print("\ncolumns / dtypes / nulls")
    print(pd.DataFrame({"dtype": df.dtypes.astype(str), "nulls": df.isna().sum()}))

    print("\ntransaction types")
    print(df["type"].value_counts())

    print("\nfraud overall     :", df["isFraud"].sum(), f"({df['isFraud'].mean():.4%})")
    print("flagged-by-rule   :", df["isFlaggedFraud"].sum(), "(isFlaggedFraud - a weak business rule, not a label)")
    print("\nfraud by type (expect fraud ONLY in TRANSFER and CASH_OUT)")
    print(df.groupby("type")["isFraud"].agg(["sum", "mean"]))

    print("\nstep (hour) range:", df["step"].min(), "->", df["step"].max())
    per_step = df.groupby("step").size()
    print("transactions per hour:\n", per_step.describe().round(1))

    print("\namount")
    print(df["amount"].describe().round(2))

    # Balance bookkeeping: does newbalance = oldbalance -/+ amount?  (Tells us how "clean" the sim is.)
    sign = df["type"].eq("CASH_IN").map({True: 1, False: -1})
    mismatch = (df["oldbalanceOrg"] + sign * df["amount"] - df["newbalanceOrig"]).abs() > 0.01
    print(f"\norigin balance does not add up: {mismatch.mean():.2%} of rows")
    print("merchant destinations (nameDest starts with M):", df["nameDest"].str.startswith("M").mean().round(3))

    fraud = df[df["isFraud"] == 1]
    print("\nfraud rows where the origin account is emptied (newbalanceOrig == 0):",
          f"{(fraud['newbalanceOrig'] == 0).mean():.1%}",
          " <- a simulator artifact; explains why fraud looks 'too easy' to detect")
    print("\nThere is NO latency, timestamp or success/failure column in PaySim.")
    print("That is exactly why we need the linkage layer with the Alibaba trace.")


def explore_alibaba():
    banner("ALIBABA MICROSERVICES v2021")
    files = callgraph_files()
    if not files:
        print("No MSCallGraph_*.csv found. Run src/simulation/download_data.py first.")
        return
    print("call-graph files:", len(files), "->", ", ".join(f.name for f in files[:4]), "...")
    print("first line of first file (header check):")
    print("  ", files[0].open(encoding="utf-8", errors="ignore").readline().strip()[:200])

    con = duckdb.connect()
    cols = open_callgraph(con)
    print("\ncolumns found:", cols)

    q = lambda sql: con.execute(sql).df()  # noqa: E731
    print("\nrows:", q("SELECT COUNT(*) AS n FROM cg")["n"][0])
    print("\ntimestamp range (ms):")
    print(q("SELECT MIN(ts) AS min_ts, MAX(ts) AS max_ts FROM cg"))
    print("\nrpctype distribution")
    print(q("SELECT rpctype, COUNT(*) AS n FROM cg GROUP BY 1 ORDER BY 2 DESC"))

    print("\nmissing / anonymised service names")
    print(q(f"""SELECT
        AVG(CASE WHEN um IS NULL OR um IN {BAD_NAMES} THEN 1 ELSE 0 END) AS um_missing,
        AVG(CASE WHEN dm IS NULL OR dm IN {BAD_NAMES} THEN 1 ELSE 0 END) AS dm_missing FROM cg"""))

    print("\nresponse time rt (ms). Negative = downstream-side record of an RPC (see README)")
    print(q("""SELECT COUNT(*) FILTER (WHERE rt < 0) AS negative, COUNT(*) FILTER (WHERE rt = 0) AS zero,
                     COUNT(*) FILTER (WHERE rt > 0) AS positive,
                     quantile_cont(ABS(rt), 0.5) AS abs_p50, quantile_cont(ABS(rt), 0.95) AS abs_p95,
                     quantile_cont(ABS(rt), 0.99) AS abs_p99, MAX(ABS(rt)) AS abs_max FROM cg"""))

    print("\ncall-graph size and depth")
    print(q("""SELECT COUNT(DISTINCT traceid) AS traces, COUNT(DISTINCT um) AS distinct_um,
                     COUNT(DISTINCT dm) AS distinct_dm FROM cg"""))
    print(q("""SELECT depth, COUNT(*) AS calls FROM
               (SELECT LENGTH(rpcid) - LENGTH(REPLACE(rpcid, '.', '')) + 1 AS depth FROM cg WHERE rpcid IS NOT NULL)
               GROUP BY 1 ORDER BY 1 LIMIT 12"""))

    extras = {p.name: len(list(p.rglob("*.csv"))) for p in ALI_RAW.iterdir() if p.is_dir()}
    print("\nextracted tables on disk (csv counts):", extras)
    print("\nNote: the call graph is a 0.5% SAMPLE of traffic, so call COUNTS are relative, not true request rates.")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "paysim"):
        explore_paysim()
    if which in ("all", "alibaba"):
        explore_alibaba()
