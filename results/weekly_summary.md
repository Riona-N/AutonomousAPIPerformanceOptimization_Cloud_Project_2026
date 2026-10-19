# Role 3 Weekly Results Summary

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
