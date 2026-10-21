# F5 Progress Report: Reproducible PPO Training Script (Riona)

## Purpose

This report covers the F5 issue assigned to Riona from the project tracker:

> "No RL training script or saved results. Add train/train_ppo.py with fixed seeds, TensorBoard logs, and saved checkpoints."

This item is about reproducibility and traceability in the RL workflow. The project needs a dedicated training script that can run the reinforcement learning job in a repeatable way, save model artifacts, log training metrics, and preserve the checkpoint history for evaluation and review.

## What is happening

The repository currently contains RL training entry points under the model package, such as:

- `src/ml_model/train_toy_agent.py`
- `src/ml_model/train_data_agent.py`

These scripts train PPO agents and save metrics or model outputs, but they are not organized under the project-level `train/` entry-point structure described by the issue. The issue specifically calls for a training script at:

- `train/train_ppo.py`

The gap is that there is no dedicated, script-based training runner under the `train/` directory with the required operational attributes:

- fixed random seeds for reproducibility,
- TensorBoard logging,
- checkpoint saving,
- and persisted output artifacts from a training run.

## Why this matters

A reinforcement learning project cannot be meaningfully reviewed or reproduced if the training process is not saved in a repeatable form. Without a dedicated script and persisted artifacts, it is difficult to:

- confirm the exact training configuration used,
- reproduce the same policy behavior across runs,
- compare results from different experiments,
- and produce verifiable evidence during professor review or team handoff.

For RL work, the training script is part of the scientific and engineering record, not just a convenience utility.

## Current project evidence

The existing training scripts show the intended direction for the project:

- they use PPO from Stable-Baselines3,
- they pass a seed into the model,
- they save model files and metrics,
- they evaluate the trained policy.

However, they are not yet organized as the required `train/train_ppo.py` workflow described in F5. In other words, the project has training logic, but not the exact reproducible script and artifact flow required by the issue.

## Issue interpretation for F5

The F5 task is not only to train a model once; it is to provide a disciplined training pipeline with:

1. fixed seeds,
2. saved checkpoints,
3. TensorBoard logs,
4. and a repeatable entry point under `train/`.

This gives the project a reliable training trail and allows further analysis of agent quality, convergence, and reproducibility.

## Expected implementation direction

The project should include a dedicated training runner similar to the following pattern:

- `train/train_ppo.py`
- sets a fixed `seed` value,
- initializes the environment and PPO model,
- logs training progress to TensorBoard,
- saves checkpoints during training,
- saves the final model artifact,
- writes evaluation metrics to a results file.

This would create a clean training pipeline aligned with the expectation in the issue.

## Fix applied and validation status

The F5 training script has been implemented and validated.

- Added the project-level training entry point at `train/train_ppo.py`.
- The script uses a fixed `--seed`, writes TensorBoard logs to `results/tensorboard/ppo`, and saves PPO checkpoints under `results/checkpoints/ppo`.
- It also saves the trained model to `results/ppo_agent.zip` and writes evaluation metrics to `results/ppo_metrics.json`.

Validation command:

```powershell
Set-Location "c:\Users\mousu\AutonomousAPIPerformanceOptimization_Cloud_Project_2026"
.\RL_env\Scripts\python.exe -m train.train_ppo --timesteps 200 --episodes 1 --checkpoint-freq 50
```

Validation result: the script completed successfully and generated training artifacts, including:

- `results/ppo_agent.zip`
- `results/ppo_metrics.json`
- `results/tensorboard/ppo/...`
- `results/checkpoints/ppo/ppo_50_steps.zip` through `ppo_250_steps.zip`

## Summary

F5 captures the missing reproducibility layer in the RL workflow. The issue states that there is no RL training script or saved results. Although the repository already contains training scripts under `src/ml_model`, the required project-level training entry point under `train/train_ppo.py` with fixed seeds, TensorBoard logging, and saved checkpoints is now in place and validated. This closes the reproducibility gap for the project and provides a reviewable training artifact trail.
