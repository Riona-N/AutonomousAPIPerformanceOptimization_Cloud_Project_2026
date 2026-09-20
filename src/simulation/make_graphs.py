"""
Draw the simulation graphs for the report into results/graphs/.

    python src/simulation/make_graphs.py

  fraud_by_type.png            fraud cases per transaction type (data section of the report)
  latency_by_stage.png         backend P50 / P95 per stage over the trace windows
  candidates_p95_by_stage.png  P95 of the 3 candidate services per stage (comparable, but they fluctuate)
  load_vs_latency.png          transactions per hour vs mean backend latency (shows the load matching works)
  policy_comparison.png        static / random / greedy routing on one simulated test day
Graphs whose input file is missing are skipped with a message.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # draw to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make src/ importable (config, common, linkage_lib)
from config import GRAPHS, OUT, STAGES  # noqa: E402
from role1_quickstart import evaluate_policies  # noqa: E402


def save(fig, name):
    fig.tight_layout()
    fig.savefig(GRAPHS / name, dpi=150)
    plt.close(fig)
    print("saved", name)


def main():
    GRAPHS.mkdir(parents=True, exist_ok=True)

    src = next((OUT / f for f in ("paysim_clean.parquet", "linkage_tx.parquet") if (OUT / f).exists()), None)
    if src is not None:
        tx = pd.read_parquet(src, columns=["type", "is_fraud"])
        by_type = tx.groupby("type")["is_fraud"].sum().reindex(["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]).fillna(0)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(by_type.index, by_type.values, color="#4472C4")
        ax.set(ylabel="Fraud cases", title="PaySim: fraud cases by transaction type")
        ax.tick_params(axis="x", rotation=20)
        save(fig, "fraud_by_type.png")
    else:
        print("skipped fraud_by_type.png (run predict.py first)")

    sw_file = OUT / "stage_latency_windows.parquet"
    if sw_file.exists():
        sw = pd.read_parquet(sw_file)
        fig, axes = plt.subplots(1, len(STAGES), figsize=(3 * len(STAGES), 3.2), sharex=True)
        for ax, s in zip(axes, STAGES):
            d = sw[(sw["stage"] == s)].groupby("window_id")[["p50", "p95"]].mean()
            ax.plot(d.index, d["p50"], label="P50"); ax.plot(d.index, d["p95"], label="P95")
            ax.set(title=s, xlabel="window (1 min)"); ax.grid(alpha=0.3)
        axes[0].set_ylabel("latency (ms)"); axes[0].legend()
        save(fig, "latency_by_stage.png")

        fig, axes = plt.subplots(1, len(STAGES), figsize=(3 * len(STAGES), 3.2), sharex=True)
        for ax, s in zip(axes, STAGES):
            for c, d in sw[sw["stage"] == s].groupby("cand_rank"):
                ax.plot(d["window_id"], d["p95"], label=f"candidate {c}")
            ax.set(title=s, xlabel="window (1 min)"); ax.grid(alpha=0.3)
        axes[0].set_ylabel("P95 latency (ms)"); axes[0].legend(fontsize=7)
        save(fig, "candidates_p95_by_stage.png")
    else:
        print("skipped latency graphs (run preprocess_alibaba.py first)")

    lk_file = OUT / "linkage_tx.parquet"
    if lk_file.exists():
        lk = pd.read_parquet(lk_file, columns=["step", "e2e_ms_static"])
        h = lk.groupby("step").agg(n=("step", "size"), latency=("e2e_ms_static", "mean"))
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.scatter(h["n"], h["latency"], s=8, alpha=0.6)
        ax.set(xlabel="Transactions in the hour", ylabel="Mean end-to-end latency, static routing (ms)",
               title="Load matching: busier hours meet a slower backend")
        ax.grid(alpha=0.3)
        save(fig, "load_vs_latency.png")
    else:
        print("skipped load_vs_latency.png (needs the full linkage_tx.parquet from build_linkage.py)")

    if (OUT / "linkage_meta.json").exists():
        res = evaluate_policies()
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        labels = [p.split(" (")[0] for p in res["policy"]]
        axes[0].bar(labels, res["mean_ms"], color="#4472C4"); axes[0].set(ylabel="Mean latency (ms)", title="Mean end-to-end latency")
        axes[1].bar(labels, res["sla_violation"] * 100, color="#ED7D31"); axes[1].set(ylabel="SLA violations (%)", title="SLA violation rate")
        for ax in axes:
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Routing policies on one simulated test day")
        save(fig, "policy_comparison.png")
    else:
        print("skipped policy_comparison.png (run build_linkage.py first)")


if __name__ == "__main__":
    main()
