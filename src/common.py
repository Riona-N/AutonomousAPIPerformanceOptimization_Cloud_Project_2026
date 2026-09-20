"""Helpers shared by the exploration and preprocessing scripts."""
import duckdb

from config import ALI_RAW

# Column order from the official README (used only if the CSV files have no header row).
README_COLS = ["timestamp", "traceid", "rpcid", "um", "rpctype", "interface", "dm", "rt"]

# Different releases spell some columns differently; map every spelling to one canonical name.
# If the explore script shows a column name we do not know, add it here.
ALIASES = {
    "timestamp": "timestamp", "ts": "timestamp",
    "traceid": "traceid", "trace_id": "traceid",
    "rpcid": "rpcid", "rpc_id": "rpcid",
    "um": "um",
    "rpctype": "rpctype", "rpc_type": "rpctype",
    "interface": "interface",
    "dm": "dm",
    "rt": "rt",
}

# Names the trace uses for "missing service name" (see README, "Missing items in traces").
BAD_NAMES = "('', '(?)', 'NAN', 'nan', 'NaN', 'None')"


def callgraph_files():
    return sorted(ALI_RAW.rglob("MSCallGraph_*.csv"))


def open_callgraph(con: "duckdb.DuckDBPyConnection"):
    """Create a DuckDB view `cg` with clean canonical columns: ts, traceid, rpcid, um, rpctype, interface, dm, rt.

    DuckDB reads the CSVs straight from disk without loading everything into RAM, which matters
    because even a few call-graph files are several GB once extracted.
    """
    files = callgraph_files()
    if not files:
        raise SystemExit("No MSCallGraph_*.csv found under data/raw/alibaba. Run 01_download.py first.")
    pattern = (ALI_RAW / "**" / "MSCallGraph_*.csv").as_posix()

    first_line = files[0].open(encoding="utf-8", errors="ignore").readline().strip()
    has_header = "traceid" in first_line.lower()
    names_arg = "" if has_header else f", names={README_COLS!r}"
    con.execute(
        f"CREATE OR REPLACE VIEW raw_cg AS SELECT * FROM read_csv('{pattern}', header={has_header}, "
        f"all_varchar=true, union_by_name=true, ignore_errors=true{names_arg})"
    )

    cols = [r[0] for r in con.execute("DESCRIBE raw_cg").fetchall()]
    found = {}
    for c in cols:
        key = ALIASES.get(c.strip().lower())
        if key and key not in found:
            found[key] = c
    missing = [k for k in README_COLS if k not in found]
    if missing:
        raise SystemExit(
            f"Could not find columns {missing}. The CSV columns are: {cols}. "
            "Add the right spelling to ALIASES in common.py."
        )

    select = ", ".join(f'"{found[k]}" AS {k}' for k in README_COLS)
    con.execute(
        f"""CREATE OR REPLACE VIEW cg AS
            SELECT TRY_CAST("timestamp" AS BIGINT) AS ts, traceid, rpcid, um,
                   lower(rpctype) AS rpctype, interface, dm, TRY_CAST(rt AS DOUBLE) AS rt
            FROM (SELECT {select} FROM raw_cg)"""
    )
    return cols
