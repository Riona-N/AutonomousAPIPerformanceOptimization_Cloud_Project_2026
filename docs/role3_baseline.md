# Role 3 — Baseline Routing Evaluation

## Objective

The purpose of the Role 3 baseline implementation is to provide
non-RL routing policies that can be used to benchmark the RL agent
developed by Role 1.

The baselines use the processed Alibaba latency-window dataset.

## Dataset Used

The baseline evaluation uses:

- `dataset/processed/stage_latency_windows.parquet`
- `dataset/processed/stage_candidates.parquet`

The latency-window dataset contains latency measurements for
candidate services across stages and time windows.

The candidate dataset contains the selected candidate services
available for each stage.

## Baseline 1 — Static Rule-Based Routing

The rule-based policy selects one candidate for each stage using
the candidate's long-run P95 latency.

For every stage:

1. Obtain all available candidate services.
2. Compare their long-run P95 latency.
3. Select the candidate with the lowest long-run P95 latency.
4. Use that candidate for the stage throughout the evaluation.

This policy does not adapt during evaluation.

## Baseline 2 — Epsilon-Greedy Bandit

The second baseline uses an epsilon-greedy multi-armed bandit.

Each candidate service is treated as an available action.

The bandit:

- explores a candidate with probability epsilon
- exploits the candidate with the highest estimated reward otherwise
- updates its estimated reward after each observation

The current implementation uses:

`epsilon = 0.1`

The reward is defined as:

`reward = -P95 latency`

Therefore, lower latency produces a higher reward.

## Evaluation Metrics

Both baselines are evaluated using the same metrics:

- Average latency
- P95 latency
- SLA violation rate
- Average reward
- Number of evaluation windows

## Evaluation Results

The evaluation was performed over 150 windows.

| Method | Average Latency (ms) | P95 Latency (ms) | SLA Violation Rate | Average Reward |
|---|---:|---:|---:|---:|
| Rule-based | 151.8714 | 347.2320 | 0% | -151.8714 |
| Epsilon-greedy Bandit | 150.8166 | 349.7776 | 0% | -150.8166 |

## Output Files

Detailed routing decisions:

`results/baseline_detailed_results.csv`

Summary comparison:

`results/baseline_comparison.csv`

## Handoff to Role 1

Role 1 should evaluate the PPO/DQN agent using the same:

- processed evaluation dataset
- latency metrics
- P95 metric
- SLA violation definition
- reward definition

This allows the RL policy to be compared against the established
non-RL baselines.
## Held-Out Test Evaluation — Phase 3

### 1. Evaluation Methodology

The baseline routing policies were evaluated using a chronological data split:

- **Training:** windows 0–20 (21 windows).
- **Validation:** windows 21–24 (4 windows).
- **Testing:** windows 25–29 (5 windows).

Five random seeds were used: 11, 22, 33, 44, and 55.

The evaluation includes rule-based routing, epsilon-greedy routing with ε = 0.05, 0.1, and 0.2, UCB1, Thompson sampling, random routing, least-loaded routing, power-of-two choices, LinUCB, and an oracle reference.

### 2. Test Results

| Routing method | Average latency (ms) | P95 latency (ms) |
|---|---:|---:|
| Epsilon-greedy (0.05) | 150.7257 | 348.0000 |
| Epsilon-greedy (0.1) | 150.7257 | 348.0000 |
| Epsilon-greedy (0.2) | 151.0260 | 348.0000 |
| Least-loaded | 151.3017 | 346.1723 |
| LinUCB | 152.6893 | 348.0000 |
| Oracle | 145.2632 | 343.4593 |
| Power of two choices | 152.1050 | 346.7540 |
| Random | 151.3402 | 347.0300 |
| Rule-based | 150.6708 | 348.0000 |
| Thompson sampling | 151.2686 | 346.1723 |
| UCB1 | 151.9388 | 345.6500 |

The recorded SLA violation rate was 0 for all methods in this evaluation.

### 3. Observations

- Epsilon-greedy with ε = 0.05 and ε = 0.1 achieved the lowest average latency among the non-oracle baselines, at 150.7257 ms.
- The rule-based baseline achieved an average latency of 150.6708 ms.
- The oracle produced the lowest average latency, 145.2632 ms. It is an optimistic reference because it selects the candidate with the lowest observed P95 latency for each test stage and window; it is not a deployable routing policy.
- The zero recorded SLA violation rate should be interpreted in the context of the current dataset and the SLA values configured for each stage.

### 4. Limitations

The held-out test set contains only five distinct time windows. The 125 recorded rows per method represent evaluations across five stages, five test windows, and five seeds; they are not 125 independent time windows.

The validation split has not been used for hyperparameter selection in this evaluation. Statistical significance and confidence intervals have not yet been calculated and should not be claimed from these results alone.

### 5. Output Files

- Detailed results: `results/baseline_test_detailed_results_v5.csv`
- Summary comparison: `results/baseline_test_comparison_v5.csv`
