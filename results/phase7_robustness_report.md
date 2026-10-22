# Phase 7 — Robustness and Sensitivity Evaluation

## 1. Linkage-seed sensitivity

Five linkage seeds were examined: [11, 22, 33, 44, 55].

The transaction-to-window mapping agreement rates for all ten seed pairs
are saved in `phase7_linkage_mapping_agreement.csv`. Static-policy outcome
summaries are saved in `phase7_linkage_seed_sensitivity.csv`.

These runs use the project's synthetic PaySim-to-Alibaba linkage. They
measure sensitivity to the linkage random seed; they do not establish
that candidate 0 is representative of every alternative routing policy.

## 2. Staleness sensitivity

The evaluator compares information delays of [0, 1, 2, 5] windows. Candidate
selection uses the stage p95 from the observed window; scoring uses the
selected candidate's p95 and SLA flag in the current window.

The delay-0 result is a **fresh-information benchmark** and is optimistic:
it uses current-window summaries to choose the candidate. It should not be
interpreted as a deployable policy result unless those measurements are
available before the routing decision.

Results are in:
- `phase7_staleness_detailed.csv`
- `phase7_staleness_summary.csv`

The summed stage p95 is a comparison proxy, not a measured end-to-end p95.
Percentiles from different stages cannot generally be added to obtain an
end-to-end percentile. The stage SLA violation fraction is the fraction
of selected stage-window records flagged as violations.

## 3. Limitations

- The chronological held-out test contains 5
  windows per delay setting.
- Linkage outcomes are generated under a static candidate-0 policy.
- The staleness experiment is a candidate-selection sensitivity test,
  not a full online deployment replay.
- Confidence claims should be limited by the small number of test windows.
- AWS/SageMaker deployment remains pending.

## 4. Reproducibility

Run from the repository root:

`python -m dataset.baselines.phase7_robustness`

Generated results are CSV files under `results/`.
