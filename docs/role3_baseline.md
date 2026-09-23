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