# D2 Robust-Agent Training Status

## Scope

This is a reduced-budget pilot of all eight D2 versions listed in
`SENTINEL-Route_Guide_Instructions.docx`, using five seeds (`0` through `4`).
The guide's PPO starting point is 1,000,000 timesteps per run; this pilot uses
2,000 PPO timesteps per seed and is preliminary, not a substitute for the full
training or final robustness evaluation.

| Version | Pilot training method |
| --- | --- |
| PPO-clean | PPO on clean observations |
| PPO-noise | PPO with uniform observation noise, epsilon 0.1 |
| PPO-PGD | PPO with ten-step PGD observation perturbations on half of observations |
| PPO-ATLA | One pilot alternation: train a learned adversary against a frozen router, then train PPO against that frozen adversary |
| PPO-trust | PPO with 15 per-stage/candidate trust features appended to the 83 base features |
| PPO-trust-ATLA | Trust-augmented router trained against a learned adversary |
| CQL | Discrete conservative Q-learning over the joint 243-action routing space |
| IQL | Discrete implicit Q-learning with expectile value fitting and advantage-weighted behavior cloning |

PPO hyperparameters follow the guide where applicable: learning rate `3e-4`,
rollout size `2048`, batch size `64`, discount `0.99`, and entropy coefficient
`0.01`. ATLA pilot sub-runs are reduced as well; they do not represent the
guide's 50,000-step alternation schedule.

## Offline-RL data limitation

No offline action-transition log or `d3rlpy` installation was present when this
pilot was prepared. CQL and IQL therefore use an in-repository discrete
implementation and collect a pilot dataset from the simulator: half of the
transitions use the fixed candidate-0 policy and half use random actions. This
is a reproducible pilot buffer, not the guide's full logged mixture of
rule-based, bandit, and random policies. Results for these versions should be
interpreted with that limitation.

## Outputs

The runner is `agents/train_d2.py`. By default, each version and seed writes to
`results/d2_preliminary/<version>/seed_<seed>/`:

- PPO model ZIP plus periodic PPO checkpoints
- CQL/IQL PyTorch checkpoints
- `curve.jsonl` learning/evaluation values
- `metrics.json`

Each version also gets an aggregate `learning_curve.png`; the run-level
`summary.json` records the selected versions, seeds, and pilot budget.

Run from the repository root:

```powershell
.\RL_env\Scripts\python.exe agents\train_d2.py `
  --timesteps 2000 `
  --seeds 0 1 2 3 4 `
  --out-dir results\d2_preliminary
```

## Execution status

**Completed: preliminary pilot, all 8 versions x 5 seeds (40 version/seed runs).**

Each version has five `metrics.json` files, five `curve.jsonl` files, and an
aggregate `learning_curve.png`. Every PPO version has a final `model.zip` plus
four periodic checkpoint files for each seed. PPO final checkpoints received a
post-training validation evaluation, which was appended to the per-seed curve
and included in the regenerated aggregate plots. CQL and IQL each have ten
PyTorch checkpoints per seed. The run manifest is
`results/d2_preliminary/summary.json`.

Verified artifacts: **30 final PPO models, 120 periodic PPO checkpoints, 10
clean-router checkpoints used to train ATLA adversaries, 100 CQL/IQL checkpoints, 40
per-version/seed metrics files, 40 curves, and 8 aggregate plots**. All
eight-by-five combinations have the expected files. The focused project test
suite passed: **14 passed**. The D2 runner completed with exit code 0.

### Interpretation and remaining work

- This is the reduced-budget pilot, not the guide's 1M-timestep PPO training.
- SB3 rounds PPO collection to a complete 2,048-step rollout for a requested
  2,000 timesteps. ATLA additionally runs one 2,048-step clean-router phase,
  one 2,048-step adversary phase, and the 2,048-step robust-router phase.
- The CQL/IQL pilot buffer contains fixed and random simulator actions only;
  it does not include bandit-policy logs.
- PPO post-training evaluation used three validation episodes per seed;
  CQL/IQL curves record two validation episodes at each logged checkpoint.
  Their return values should therefore not be compared as if they used identical
  evaluation sampling.
- This pilot does not establish statistical robustness, non-overlapping
  confidence intervals, attack performance, or shield recovery. Those require
  longer training and evaluation across clean and attacked validation runs.
- ATLA's final router and its clean router used to train the adversary are
  checkpointed. The learned adversary policy itself is not currently saved as
  a separate artifact.

For additional or full-budget runs, use the command above. The runner skips
already-completed version/seed outputs when their metrics match the requested
timesteps, so a completed pilot can be resumed or extended with a different
output directory.
