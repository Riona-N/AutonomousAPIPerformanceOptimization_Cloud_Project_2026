# Role 1: Data-backed RL baseline

## State and action contract

`DataApiRoutingEnv` uses one routing decision per transaction across the five
stages `ingress`, `auth`, `routing`, `processor`, and `db`.

- Observation: 83 float32 values in `[0, 1]`.
- Per stage and candidate: `p50`, `p95`, `p99`, `log1p(n_calls)`, and the SLA flag.
- The metric block has `5 stages x 3 candidates x 5 metrics = 75` values.
- Transaction block: five one-hot type flags, normalized log amount, fraud risk,
  and transfer flag (`8` values).
- Metrics are read from the previous minute's trace window, while the current
  transaction's window supplies the sampled latency.
- Action: `MultiDiscrete([3, 3, 3, 3, 3])`; each value selects candidate `0`,
  `1`, or `2` for the corresponding stage.

Scaling and prioritization are intentionally not actions: Role 3's data does
not contain a measured effect for them. They can be added after Role 2 supplies
an execution model or AWS measurements.

## Reward

For a sampled end-to-end latency `L`, the shared thresholds from
`linkage_meta.json` are used:

```text
reward = success_bonus - L / sla_ms - 0.75 * sla_violation
         + 0.20 * (1 - fraud_risk)
```

`success_bonus` is `+1` when `L <= timeout_ms`, otherwise `-1`. Success and
latency are simulated from the empirical Role 3 quantile tables, so results are
relative comparisons on the same backend simulation, not production claims.

## Training

The runner uses the committed sample automatically and falls back to the full
`linkage_tx.parquet` when it is generated:

```powershell
./RL_env/Scripts/python.exe -m src.ml_model.train_data_agent --timesteps 10000
```

The model is written to `results/data_ppo_agent.zip` and validation metrics to
`results/data_ppo_metrics.json`. The runner trains on `split == "train"` and
evaluates on `split == "val"`; the held-out `test` split remains available for
the final comparison.

Load and call the saved model:

```python
from stable_baselines3 import PPO
from src.ml_model.data_api_env import DataApiRoutingEnv

env = DataApiRoutingEnv(split="test")
model = PPO.load("results/data_ppo_agent", env=env)
observation, _ = env.reset()
action, _ = model.predict(observation, deterministic=True)
```