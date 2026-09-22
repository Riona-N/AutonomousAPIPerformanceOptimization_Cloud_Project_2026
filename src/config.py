"""Shared settings for the Role 3 data pipeline. Edit values here, not inside the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # repo root: contains dataset/, src/, results/, docs/
DATASET = ROOT / "dataset"
RAW = DATASET / "raw"
PAYSIM_RAW = RAW / "paysim"
ALI_RAW = RAW / "alibaba"
OUT = DATASET / "processed"                     # small outputs are committed, big ones are regenerated
RESULTS = ROOT / "results"
GRAPHS = RESULTS / "graphs"
MODEL_PATH = ROOT / "src" / "ml_model" / "model.pkl"

SEED = 42

# ---------------- PaySim ----------------
# step = 1 hour, steps 1..743. We split by TIME (never randomly) so the test set is "the future".
TRAIN_MAX_STEP = 520   # ~70% of the timeline
VAL_MAX_STEP = 631     # ~15%; everything after this is test

# ---------------- Alibaba ----------------
WINDOW_MS = 60_000     # one simulation window = 60 s of trace time
STAGES = ["ingress", "auth", "routing", "processor", "db"]   # the payment-gateway call chain
CANDIDATES_PER_STAGE = 3   # routable "replicas" per stage (rank 0 = busiest of the chosen three)
CANDIDATE_POOL = 10        # look at the 10 busiest services per stage, then pick the 3 that are closest in speed
MIN_COVERAGE = 0.8         # a service must have enough calls in >= 80% of windows to be eligible
EQUALIZE_CANDIDATES = True # rescale the chosen 3 so their long-run P95 level is identical (see data_dictionary.md)
MAX_RT_MS = 60_000     # drop absurd response times above this (bad records)
MIN_CALLS_PER_CELL = 5 # a (stage, candidate, window) cell with fewer calls is imputed, not trusted

# ---------------- Definitions the WHOLE TEAM must agree on ----------------
# Role 1 uses these in the reward; you use them in the evaluation. Change once, here.
SLA_MULT = 2.0       # SLA violation  if end-to-end latency > SLA_MULT     x baseline median
TIMEOUT_MULT = 4.0   # transaction FAILS if end-to-end latency > TIMEOUT_MULT x baseline median
