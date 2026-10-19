# Phase 6 — Baseline Evaluation Report

## Evaluation setup

Eleven routing methods were evaluated over five random seeds and five chronological held-out test windows. Metrics include mean latency, P95 and P99 latency, SLA violation rate, average reward, and 95% confidence intervals.

The confidence intervals use the Student-t distribution across seed-level results. The validation split was not used for hyperparameter tuning in this evaluation.

## Results

The rule-based policy achieved the lowest mean latency among the evaluated non-oracle methods at **150.67 ms**. Epsilon-greedy with exploration rates 0.05 and 0.1 achieved approximately **150.73 ms**. The oracle reference achieved 145.26 ms but represents an optimistic benchmark, not a deployable routing policy.

All evaluated methods recorded zero SLA violations across the observed test windows. This result applies only to the current evaluation sample.

## Limitations

- Only five held-out windows were evaluated, limiting conclusions about generalization to future workloads.
- Confidence intervals are calculated across five seed-level aggregates, not independent workload windows.
- Zero-width intervals for deterministic policies indicate identical observed results across seeds; they do not establish certainty about future performance.
- The PaySim–Alibaba linkage uses a documented synthetic mapping. Candidate-0 latency outcomes must not be treated as observed counterfactual outcomes for other routing candidates.
- AWS/SageMaker deployment remains pending.

## Conclusion

The rule-based policy had the best observed mean latency among non-oracle methods in this test. Epsilon-greedy at 0.05 and 0.1 performed similarly. More held-out windows and robustness experiments are required before making stronger claims about policy superiority or deployment readiness.