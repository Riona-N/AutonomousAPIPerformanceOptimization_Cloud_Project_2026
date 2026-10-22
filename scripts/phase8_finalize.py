from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
P = R / "plots"
P.mkdir(parents=True, exist_ok=True)

def newest(pattern):
    files = list(R.glob(pattern))
    return max(files, key=lambda f: f.stat().st_mtime) if files else None

def load(path):
    return pd.read_csv(path) if path else None

def pick(df, words, numeric_only=False):
    if df is None:
        return None
    cols = df.select_dtypes(include="number").columns if numeric_only else df.columns
    return next((c for c in cols if any(w in c.lower() for w in words)), None)

comparison_path = newest("baseline_test_comparison_*.csv")
detailed_path = newest("baseline_test_detailed_results_*.csv")
if not comparison_path:
    raise SystemExit("No baseline_test_comparison_*.csv found in results/")

tables = []
for label, pattern in [
    ("baseline_comparison", "baseline_test_comparison_*.csv"),
    ("baseline_detailed", "baseline_test_detailed_results_*.csv"),
    ("linkage_seed_sensitivity", "phase7_linkage_seed_sensitivity.csv"),
    ("linkage_mapping_agreement", "phase7_linkage_mapping_agreement.csv"),
    ("staleness_summary", "phase7_staleness_summary.csv"),
    ("staleness_detailed", "phase7_staleness_detailed.csv"),
]:
    path = newest(pattern) if "*" in pattern else (R / pattern if (R / pattern).exists() else None)
    if path:
        df = pd.read_csv(path)
        df.insert(0, "source_table", label)
        df.insert(1, "source_file", path.name)
        tables.append(df)

pd.concat(tables, ignore_index=True, sort=False).to_csv(R / "summary.csv", index=False)

df = load(comparison_path)
method = pick(df, ["method", "policy", "baseline", "algorithm"])
metric = pick(df, ["mean_latency", "avg_latency", "latency_mean", "mean_ms", "avg_ms"], True)
if method and metric:
    plot = df[[method, metric]].dropna().sort_values(metric)
    if not plot.empty:
        plt.figure(figsize=(9, 5))
        plt.barh(plot[method].astype(str), plot[metric])
        plt.xlabel(metric)
        plt.title("Held-out baseline comparison")
        plt.tight_layout()
        plt.savefig(P / "baseline_comparison.png", dpi=160)
        plt.close()
else:
    print("WARNING: baseline plot skipped; inspect CSV column names.")

seed_path = R / "phase7_linkage_seed_sensitivity.csv"
if seed_path.exists():
    df = pd.read_csv(seed_path)
    x = pick(df, ["seed"])
    y = pick(df, ["mean_e2e", "mean_latency", "p95"], True)
    if x and y:
        plt.figure(figsize=(8, 4.5))
        plt.plot(df[x], df[y], marker="o")
        plt.xlabel(x); plt.ylabel(y); plt.title("Linkage seed sensitivity")
        plt.tight_layout(); plt.savefig(P / "linkage_seed_sensitivity.png", dpi=160); plt.close()
    else:
        print("WARNING: linkage plot skipped; inspect CSV column names.")

stale_path = R / "phase7_staleness_summary.csv"
if stale_path.exists():
    df = pd.read_csv(stale_path)
    x = pick(df, ["delay", "staleness"])
    y = pick(df, ["stage_p95", "mean_p95", "latency", "sla"], True)
    if x and y:
        plt.figure(figsize=(8, 4.5))
        plt.plot(df[x], df[y], marker="o")
        plt.xlabel(x); plt.ylabel(y); plt.title("Observation staleness sensitivity")
        plt.tight_layout(); plt.savefig(P / "staleness_sensitivity.png", dpi=160); plt.close()
    else:
        print("WARNING: staleness plot skipped; inspect CSV column names.")

(R / "weekly_summary.md").write_text("""# Role 3 Weekly Results Summary

## Completed
- Chronological Alibaba train/validation/test split and PaySim–Alibaba linkage documentation.
- Reusable fraud-risk function in `data/risk.py`.
- Routing baseline evaluation and held-out result exports.
- Offline-RL transition data and schema exports.
- Five-seed linkage sensitivity experiment.
- Observation-delay study for 0, 1, 2 and 5 windows.

## Outputs
- `results/summary.csv`
- `results/plots/baseline_comparison.png` (if columns were identifiable)
- `results/plots/linkage_seed_sensitivity.png` (if columns were identifiable)
- `results/plots/staleness_sensitivity.png` (if columns were identifiable)

## Limitations and handoff
- The held-out Alibaba test partition has five windows; findings are preliminary.
- Oracle is an optimistic reference, not a deployable policy.
- Linkage sensitivity is based on simulated mappings.
- Summed stage p95 is a comparison proxy, not measured end-to-end p95.
- Stage-level SLA flags do not establish end-to-end SLA compliance.
- AWS/SageMaker deployment remains pending.
- Confirm Riona received and can use the offline-RL data; confirm `fraud_risk` integration with A5.
- Full attack-matrix and end-to-end integration remain unverified.

## Reproduction
From the repository root, run `python scripts/phase8_finalize.py`.
This regenerates summaries and plots from existing CSVs; it does not rerun experiments.
""")

changelog = ROOT / "CHANGELOG.md"
entry = "\n## Phase 8 — Role 3 reporting\n- Added consolidated results summary and robustness/baseline plots.\n- Added weekly results summary and reproduction instructions.\n- AWS/SageMaker and external integration remain pending until verified.\n"
if changelog.exists():
    old = changelog.read_text()
    if "Phase 8 — Role 3 reporting" not in old:
        changelog.write_text(old.rstrip() + "\n" + entry)
else:
    changelog.write_text("# Changelog\n" + entry)

print("Phase 8 outputs generated.")
print("Review results/summary.csv and results/plots before committing.")
