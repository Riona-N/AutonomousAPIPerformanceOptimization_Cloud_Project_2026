"""
LatencyModel - the interface between Role 3 (data) and Role 1 (RL environment).

    from linkage_lib import LatencyModel
    lm = LatencyModel("data/processed/stage_latency_windows.parquet")
    ms = lm.sample(stage_idx=2, cand=1, window_ids=[17, 17, 18], rng=rng)   # ms of latency for 3 requests

"If the agent routes this request to candidate `cand` of stage `stage_idx` while the backend is in
trace window `window_id`, how long does the call take?"  Answered by inverse-CDF sampling from the
REAL response-time quantiles of that service in that window (no distribution assumed: real latencies
are a lump of near-zero calls plus a long tail, which a lognormal fits badly).
"""
import numpy as np
import pandas as pd

try:
    from config import STAGES
except ImportError:
    from src.config import STAGES

# Quantile grid stored per (stage, candidate, window). A uniform draw u is mapped through it by linear interpolation.
QUANTILE_LEVELS = np.array([0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999])
QCOLS = ["q00", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99", "q999"]


class LatencyModel:
    def __init__(self, path):
        df = pd.read_parquet(path)
        self.stages = STAGES
        self.windows = np.sort(df["window_id"].unique())
        n_cand = int(df["cand_rank"].max()) + 1
        self.q = np.full((len(STAGES), n_cand, len(self.windows), len(QCOLS)), np.nan, np.float32)
        si = df["stage"].map({s: i for i, s in enumerate(STAGES)}).to_numpy()
        wi = np.searchsorted(self.windows, df["window_id"].to_numpy())
        self.q[si, df["cand_rank"].to_numpy(), wi] = df[QCOLS].to_numpy()
        self.p95 = self.q[..., QCOLS.index("q95")]                 # shape (stage, cand, window)
        self.n_cand = (~np.isnan(self.p95[:, :, 0])).sum(axis=1)   # candidates that exist per stage

    def pos(self, window_ids):
        """window_id (as in the trace) -> array position."""
        return np.searchsorted(self.windows, np.asarray(window_ids))

    def sample(self, stage_idx, cand, window_ids, rng=None, u=None, chunk=500_000):
        """Draw one latency (ms) per element of window_ids. `cand` may be a scalar or an array of the same length.
        Pass `u` (values in 0..1) instead of `rng` to get deterministic quantiles, e.g. u=0.95 returns the stored P95."""
        p = self.pos(window_ids)
        n, k = len(p), len(QUANTILE_LEVELS)
        out = np.empty(n, np.float32)
        u_all = None if u is None else np.asarray(u, dtype=float)
        for a in range(0, n, chunk):                     # chunked so 6M transactions do not need GBs of RAM
            pp = p[a:a + chunk]
            c = cand[a:a + chunk] if np.ndim(cand) else cand
            uu = rng.random(len(pp)) if u_all is None else u_all[a:a + chunk]
            i = np.clip(np.searchsorted(QUANTILE_LEVELS, uu, side="right") - 1, 0, k - 2)
            frac = np.clip((uu - QUANTILE_LEVELS[i]) / (QUANTILE_LEVELS[i + 1] - QUANTILE_LEVELS[i]), 0, 1)
            rows = self.q[stage_idx, c, pp]              # (m, k) quantile rows for each request's cell
            r = np.arange(len(pp))
            out[a:a + chunk] = rows[r, i] * (1 - frac) + rows[r, i + 1] * frac
        return out

    def stress_by_window(self):
        """How slow is the static (rank-0) path in each window? Sum of the stage P95s. Used for load matching."""
        return self.p95[:, 0, :].sum(axis=0)
