"""
Shows what SLA-violation and failure rates the static baseline gets for different threshold multipliers.
Read-only: it changes nothing. Pick values, edit config.py, then re-run src/simulation/build_linkage.py and src/simulation/validate.py.

  python src/simulation/tune_thresholds.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make src/ importable (config, common, linkage_lib)
import numpy as np
import pandas as pd

from config import OUT, SLA_MULT, TIMEOUT_MULT

e2e = pd.read_parquet(OUT / "linkage_tx.parquet", columns=["e2e_ms_static"])["e2e_ms_static"].to_numpy()
med = float(np.median(e2e))

print(f"static-baseline end-to-end latency: median {med:.0f} ms")
print("  " + "  ".join(f"P{p}: {np.percentile(e2e, p):.0f}" for p in (90, 95, 99, 99.9)) + "  (ms)")
print(f"\ncurrent config: SLA_MULT={SLA_MULT}, TIMEOUT_MULT={TIMEOUT_MULT}")
print(f"  SLA violation {np.mean(e2e > SLA_MULT * med):.2%}   failures {np.mean(e2e > TIMEOUT_MULT * med):.2%}")

print("\nSLA_MULT -> share of transactions violating the SLA   (aim for roughly 5-15%)")
for m in (1.5, 2, 2.5, 3, 4, 5):
    print(f"  {m:>4}x median = {m * med:7.0f} ms   violations {np.mean(e2e > m * med):7.2%}")

print("\nTIMEOUT_MULT -> share of transactions that fail        (aim for roughly 1-5%)")
for m in (3, 4, 5, 6, 8, 10, 15):
    print(f"  {m:>4}x median = {m * med:7.0f} ms   failures   {np.mean(e2e > m * med):7.2%}   success {np.mean(e2e <= m * med):.2%}")
