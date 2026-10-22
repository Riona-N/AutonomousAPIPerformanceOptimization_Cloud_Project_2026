# Attack Implementation Report

## Scope

This report documents the adversarial attack wrappers implemented for the routing environment, based on the attack specification in `SENTINEL-Route_Guide_Instructions.docx`.

The requested work includes:

- `AttackWrapper` base class
- A1 telemetry poisoning
- A2 observation perturbation with random, FGSM, PGD, and learned PPO variants
- A5 risk-score poisoning
- A6 metric replay

## What was implemented

### 1) Base wrapper structure

The project now includes a reusable adversarial wrapper package in `attacks/`:

- `attacks/base.py`
- `attacks/wrappers.py`
- `attacks/__init__.py`

The core abstraction is `AttackWrapper`, which follows the design described in the guide:

- wraps a Gymnasium environment without modifying the environment itself,
- tracks the step counter and attack window,
- exposes `active()` based on `start` and `duration`,
- keeps `true_obs` and `seen_obs` in the returned info payload,
- stores attack metadata such as `attack_id` and `attack_active`.

This makes all attacks consistent with the project requirement: every attack is a wrapper that modifies only the observation seen by the policy while keeping the base environment behavior unchanged.

### 2) A1 — Telemetry poisoning

Implemented via `TelemetryPoison`.

This attack simulates a malicious candidate that is made to appear better than it really is by altering the observed metrics for selected stages. The logic follows the guide:

- choose a subset of stages according to `rho`
- identify the attacker candidate with the worst long-run p95 for each selected stage
- identify the best candidate for the same stage
- overwrite the attack candidate's observed metrics to look like the best candidate minus a configurable margin
- zero the SLA flag for the attacker candidate to hide suspicious behavior

### 3) A2 — Observation perturbation

Implemented as three variants:

- `RandomObservationAttack`
- `FGSMObservationAttack`
- `PGDObservationAttack`

These attacks follow the attack specification in the guide:

- random perturbation: bounded uniform noise within `epsilon`
- FGSM: one-step sign attack using a surrogate gradient from the policy
- PGD: iterative projected gradient attack with 10 steps and step-size `epsilon / 4`
- learned adversary: a second PPO policy outputs an observation-sized perturbation vector bounded by `epsilon`; it is trained against a frozen routing policy with reward equal to the negative of the router's reward

FGSM and PGD fall back to random perturbation if no surrogate policy is available. The learned PPO wrapper instead requires its trained policy and raises a clear error if it is missing.

#### Learned A2 adversary training

The learned attack implementation adds:

- `LearnedAdversaryEnv`, which exposes the clean routing observation to the adversary and bounds its continuous action to `[-epsilon, epsilon]` per feature.
- `train/train_adversary_ppo.py`, which loads the existing router PPO artifact once, uses it only for deterministic action selection, and trains a separate PPO model to maximize the negative router reward.
- `LearnedAdversaryAttack`, which loads as an `AttackWrapper` policy through `AttackConfig.policy` and applies its bounded perturbation to the observation seen by the router.

Train the adversary from the repository root with:

```powershell
.\RL_env\Scripts\python.exe -m train.train_adversary_ppo `
  --router-path results\ppo_agent.zip `
  --output-path results\adversary_ppo `
  --timesteps 50000 `
  --epsilon 0.1
```

The router artifact is not updated during adversary training. The adversary artifact is saved as `results/adversary_ppo.zip` by default. At inference, load it with `PPO.load(...)` and pass it as the `policy` in an A2 `AttackConfig` alongside the same `epsilon`.

### 4) A5 — Risk-score poisoning via the project risk signal

Implemented via `RiskAttack`.

This wrapper treats the fraud-risk score as a first-class attack surface. The implementation tries to load a project-level risk function when available, and otherwise falls back to the environment's current fraud-risk value plus a bounded estimate derived from the transaction amount and type. While active, the wrapper rewrites the observation's risk feature toward a higher malicious value using a configurable `risk_bias`, so the policy sees a riskier transaction than the true signal indicates.

This matches the project pattern of keeping the actual environment unchanged while modifying only what the policy observes. It is therefore consistent with the wrapper-based attack design used for A1, A2, and A6.

### 5) A6 — Metric replay

Implemented via `MetricReplayAttack`.

This attack keeps a short observation history and, while active, replays a metric block from an earlier healthy window using the configured `delta` offset. This matches the guide's description of replaying earlier latency blocks to hide real degradation or outages.

## Validation status

The attack logic was validated with a focused pytest run covering the environment, attack wrappers, and training smoke tests. The learned-adversary tests check the epsilon bound, adversarial reward sign, and inference-wrapper perturbation bound. The training entry point was also run for 8 timesteps against the saved router artifact and successfully wrote an adversary checkpoint in a temporary directory. That checkpoint was loaded through `LearnedAdversaryAttack`; inference passed with an observation shape of 83 and a maximum perturbation of approximately 0.0622 for `epsilon=0.1`.

Command run:

```powershell
Set-Location "c:\Users\mousu\AutonomousAPIPerformanceOptimization_Cloud_Project_2026"
.\RL_env\Scripts\python.exe -m pytest tests\test_attacks.py tests\test_data_api_env.py tests\test_train_ppo.py -q
```

Result:

- 14 passed

## Files added/updated

- `attacks/__init__.py`
- `attacks/adversary.py`
- `attacks/base.py`
- `attacks/wrappers.py`
- `train/train_adversary_ppo.py`
- `tests/test_attacks.py`
- `docs/ATTACKS_IMPLEMENTATION_REPORT.md`

## Notes

This implementation focuses on the simulator-side attack wrappers required by the project specification. The AWS sandbox or defense-layer components are separate follow-on work and were not added in this pass.
