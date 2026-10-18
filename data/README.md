# PaySim–Alibaba Data Linkage (F10)

## Purpose

PaySim provides synthetic financial transaction traffic, while Alibaba
provides backend service latency measurements. The datasets do not share
a common event timestamp or transaction identifier suitable for a direct
join. The project therefore uses an explicit synthetic load-to-stress
mapping rather than claiming a real timestamp-based join.

Implementation: `src/simulation/build_linkage.py`
Latency model: `src/linkage_lib.py`
Alibaba preprocessing: `src/simulation/preprocess_alibaba.py`

## Rule A — Assign PaySim traffic to Alibaba latency windows

1. Read the cleaned PaySim transactions, including `step` and `ts_sec`.
2. Count transactions per PaySim `step`, treating each step as an hour.
   The hourly count series is reindexed to consecutive steps beginning at 1.
3. Calculate the percentile rank of each hourly transaction count.
   Missing hourly counts receive the middle percentile (0.5).
4. Calculate Alibaba backend stress for each latency window as the sum
   of candidate 0's P95 latency across all configured service stages.
5. Sort Alibaba windows from lowest stress to highest stress.
6. Convert each PaySim hourly traffic percentile into a base position
   in this sorted stress-window list.
7. Convert each transaction timestamp to a minute slot using
   `slot = floor(ts_sec / 60)`. Each minute slot receives one Alibaba
   window. The base position is adjusted by seeded random jitter and
   clipped to the available range.
8. Assign every transaction in that minute slot the same `window_id`.
   Save the slot-to-window mapping as `slot_to_window.parquet`.

The random seed controls the jitter. Different seeds can produce
different mappings. This is a synthetic relationship: busier PaySim
hours are associated with more stressed Alibaba windows, but this does
not establish that the two datasets describe the same real-world events.

## Rule B — Generate transaction latency

For each transaction, the implementation samples a latency for every
configured stage using the inverse-CDF quantile model in `LatencyModel`.
The selected window is the transaction's assigned `window_id`, and the
candidate is fixed to rank 0 for all stages in this linkage-generation
step.

The sampled stage latencies are summed to calculate end-to-end latency.
The output includes the generated latency, SLA-violation indicator, and
success indicator. The generated transaction data and run metadata are
saved under the seed-specific linkage output directory.

## SLA and timeout threshold caveat

In the current implementation, the median end-to-end latency is calculated
from the generated transaction set for that run. The SLA threshold is
`SLA_MULT * median`, and the timeout threshold is
`TIMEOUT_MULT * median`.

These are simulation-derived thresholds, not independently specified
production SLAs. Because the median is calculated from the generated
transaction set, results should be interpreted with this limitation in
mind.

## Chronological baseline evaluation (F9)

`dataset/baselines/time_split.py` splits unique Alibaba `window_id` values
in chronological order. All rows belonging to a window remain in the same
split; windows are not randomly shuffled.

With the current 30 unique windows and default ratios, the split is:
- Training: windows 0–20 (21 windows)
- Validation: windows 21–24 (4 windows)
- Test: windows 25–29 (5 windows)

The exact counts depend on the number of unique windows and the split
implementation's integer boundaries.

The baseline evaluator runs five seeds: 11, 22, 33, 44, and 55. Online
bandit methods select a candidate and then receive the observed latency
for that test window as feedback. Their results therefore represent
online adaptation on the chronological test sequence, not a fully
frozen policy. The oracle uses observed test-window outcomes to choose
the best candidate and is an optimistic reference, not a deployable policy.

## Benchmark-pool limitation

`src/simulation/preprocess_alibaba.py` selects candidate services using
statistics calculated across the available Alibaba trace, including
latency levels and window coverage. The chronological split is applied
later by the baseline evaluator.

Consequently, the current split prevents baseline policy fitting from
using future-window outcomes directly, but candidate-pool construction
has already used full-trace information. This should be disclosed when
interpreting test results; the experiment is not a fully independent
evaluation of candidate discovery.

## Reproducibility and outputs

The linkage-generation seed determines the random minute-slot jitter and
latency sampling. The baseline evaluator uses its configured seed list
and writes timestamped detailed and summary CSV files under `results/`.

Keep the seed, configuration, generated linkage metadata, and result
filenames with any reported metrics so that runs can be traced and
compared.
EOF