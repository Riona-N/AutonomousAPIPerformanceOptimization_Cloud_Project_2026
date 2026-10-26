# D4 Extended Reward and Validation Grid Search

## Reward implementation

`DataApiRoutingEnv` now accepts three optional reward weights:

- `reward_beta`: cost penalty weight, `beta`
- `reward_gamma`: low-trust routing penalty weight, `gamma`
- `reward_delta`: low-risk fraud-value penalty weight, `delta`

The environment preserves the existing reward when all weights are zero. Each
step reports the base reward and each D4 term under `info["reward_terms"]`.

The current repository has no candidate/transaction cost field. Following the
requested scope, `cost_t` is therefore zero until real cost data is supplied.
Consequently, beta cannot affect training or be identified by this grid search.

The existing environment does not consume D1 tomography trust scores. For the
interim low-trust penalty, the chosen candidate's trust is estimated from the
existing observation metrics as normalized call support multiplied by SLA
compliance; the per-transaction penalty input is the fraction of stages whose
chosen candidate has estimated trust below `theta = 0.5`. This is a documented
proxy, not the guide's calibrated D1 trust score.

When the transaction row has a ground-truth `is_fraud` label, the fraud-value
term is the transaction amount normalized to `[0, 1]`, multiplied by the fraud
label and applied only when predicted `fraud_risk < theta`. If the label is
absent or missing, this term is zero.

## Grid-search procedure

`eval/grid_search_d4.py` trains one PPO model per weight combination using the
train split and tunes/evaluates the grid exclusively on the validation split.
The test split is not loaded. The pilot uses three values per weight:

- `beta`: `0`, `0.1`, `1.0`
- `gamma`: `0`, `0.5`, `1.0`
- `delta`: `0`, `0.5`, `1.0`

This yields 27 combinations per seed. Defaults are 2,048 requested PPO
timesteps and one seed. Two validation episodes are used for this execution.
PPO may collect a complete rollout even if a smaller timestep budget is passed.

The selection score is reported separately from the trained D4 reward:
`mean base reward - mean low-trust fraction - mean normalized fraud value`.
It uses fixed unit weights so candidates are compared under a common validation
criterion rather than their own coefficient-weighted rewards. This is a
preliminary selection heuristic, not a claim of statistically optimal weights.

## Artifacts

The runner writes:

- one PPO checkpoint and `validation_metrics.json` per seed/weight combination,
- `grid_results.csv` with all validation results and reward components,
- `summary.json` with protocol, selected weights, and cost identifiability,
- `grid_search.png` showing the validation selection score.

Run from the repository root:

```powershell
.\RL_env\Scripts\python.exe eval\grid_search_d4.py `
  --timesteps 2048 `
  --seeds 42 `
  --episodes 2 `
  --out-dir results\d4_grid
```

## Validation and status

The environment test suite covers default reward compatibility, low-trust and
fraud-label penalties, and invalid configuration inputs. The full D4 grid is
complete: **27 combinations**, one seed (`42`), and 2,048 PPO timesteps per
combination. The models were trained on `train`; tuning metrics use `val` only.
Artifacts include 27 PPO checkpoints, 27 validation metric files,
`results/d4_grid/grid_results.csv`, `summary.json`, and `grid_search.png`.

The validation grid found **no non-zero penalty observations** for any
coefficient: `mean_cost = 0`, `mean_low_trust_fraction = 0`, and
`mean_normalized_fraud_value = 0` across all runs. Thus all 27 runs had the same
mean episode return (`194.7573`), and none of `beta`, `gamma`, or `delta` is
identifiable from this validation sample. The manifest consequently records
`selected_weights: null` rather than claiming a tuned optimum. This is the
result of the grid search: the reward terms exist and are tested, but required
cost and low-trust/low-risk events are not represented in the current
validation rollouts.

Validation commands:

```powershell
.\RL_env\Scripts\python.exe -m pytest tests\test_data_api_env.py -q
.\RL_env\Scripts\python.exe -m pytest tests\test_shield.py tests\test_attacks.py tests\test_data_api_env.py -q
```

The D4-specific environment tests passed (**10 passed**); the combined shield,
attack, and environment suite passed (**21 passed**). Pytest emitted its known
cache-path warning, but no test failed.

## Limitations

- Beta is not identifiable until a real candidate cost signal/provider exists.
- The gamma term uses an interim telemetry proxy; D1 trust-layer implementation
  and clean-validation calibration are needed for the specified trust score.
- The current validation sample generated no low-trust selections below
  `theta = 0.5` and no labeled fraud with `fraud_risk < 0.5`, so gamma/delta
  also require a more informative validation scenario to tune.
- Grid selection uses a small validation sample and one seed. It is preliminary,
  and the test split must remain reserved for final reporting.
