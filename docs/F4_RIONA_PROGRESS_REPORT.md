# F4 Progress Report: Candidate Count / Observation Size Alignment (Riona)

## Purpose

This report covers the F4 issue assigned to Riona from the project tracker:

> "Candidate count fixed at 3, stage count from config; observation size becomes stages × K × 5 + 8."

The issue is about a mismatch between the configuration values and the environment's actual observation shape. The project defines the stage list in configuration, but the environment still hard-codes the candidate count per stage to 3. That creates a consistency problem: the model may describe candidate counts in one place and then silently use a different shape in the RL environment.

## What is happening

The configuration file defines both the stage list and the candidate count:

- `src/config.py` sets:
  - `STAGES = ["ingress", "auth", "routing", "processor", "db"]`
  - `CANDIDATES_PER_STAGE = 3`

At the same time, the environment currently initializes its internal values as:

```python
stages = tuple(STAGES)
candidates_per_stage = 3
metrics_per_candidate = 5
```

This means the stage count comes from configuration, but the candidate count is still effectively fixed in the environment, even though the configuration already exposes a canonical value.

## Why this matters

The observation layout is built from:

```python
len(self.stages) * self.candidates_per_stage * self.metrics_per_candidate
+ len(self.transaction_types) + 3
```

For the current project,

- stage count = 5
- candidate count = 3
- metrics per candidate = 5
- transaction-type flags = 5
- extra transaction features = 3

So the observation size becomes:

```text
5 × 3 × 5 + 5 + 3 = 75 + 8 = 83
```

This is the intended arrangement described in the issue: `stages × K × 5 + 8` where `K` is the candidate count per stage.

The real problem is that the environment is not reading `K` dynamically from configuration. That makes the system brittle if the configured number of candidates changes later, and it makes the observation shape drift away from the shared data contract.

## Current implementation status

The project currently shows the following behavior:

- `STAGES` is loaded from `src/config.py` and used as the active stage dimension.
- candidate count is still a hard-coded environment constant instead of being pulled from shared config.
- the observation space is derived from the hard-coded candidate count, so the environment is effectively tied to a fixed value of 3.

This means the F4 issue is present in the implementation: the stage count is config-driven, but the candidate count is not yet fully config-driven.

## Why this is a problem for RL correctness

The observation tensor defines the state representation that the policy consumes. If the candidate count and the observation size are not consistently tied to configuration, then the following can happen:

- the observation shape may be valid for one training run and invalid for another,
- the code may silently assume a fixed number of candidates even when the dataset/config says otherwise,
- the agent policy and the environment feature vector can become misaligned,
- evaluation and inference may break if the candidate layout changes.

For a robust RL setup, the environment's feature vector should be derived directly from the same configured values used elsewhere in the pipeline.

## Expected fix direction

The intended fix is to keep the stage count from config and also read the candidate-per-stage count from the same shared config instead of hard-coding it in the environment. That would make the observation size calculation consistent with the project-wide data model:

```python
self.stages = tuple(STAGES)
self.candidates_per_stage = CANDIDATES_PER_STAGE
```

Then the shape calculation automatically follows the shared configuration values, e.g.:

```python
observation_size = (
    len(self.stages) * self.candidates_per_stage * self.metrics_per_candidate
    + len(self.transaction_types)
    + 3
)
```

This keeps the environment and config in sync.

## Fix applied and validation status

The F4 fix has been implemented and verified.

- The environment now imports `CANDIDATES_PER_STAGE` from `src/config.py` and uses it directly in `src/ml_model/data_api_env.py`.
- The environment keeps `self.candidates_per_stage = int(CANDIDATES_PER_STAGE)` so the state representation stays aligned with the shared project configuration.
- A regression test was added to confirm that the observation-space shape matches the configured stage/candidate layout.

Validation command:

```powershell
Set-Location "c:\Users\mousu\AutonomousAPIPerformanceOptimization_Cloud_Project_2026"
.\RL_env\Scripts\python.exe -m pytest tests\test_data_api_env.py -q
```

Result: 6 tests passed.

## Summary

F4 is about the configuration-to-environment mismatch: the environment uses the stage count from config but still hard-codes the candidate count to 3. The observation size therefore depends on a fixed value rather than the project configuration. This is a structural consistency issue in the RL state representation, and it needs to be resolved by reading the candidate count from the shared configuration instead of leaving it hard-coded.
