# D3 Safety Shield with Hysteresis

## Implementation

Added `defense/shield.py` with two interfaces:

- `SafetyShield.apply(...)` is a policy-independent decision function. It receives
  one candidate choice per stage, stage/candidate trust scores, and level p95
  values.
- `SafetyShieldWrapper` applies the shield after a policy chooses an action and
  records overrides and alerts in the environment `info` dictionary.

For each stage, a proposed candidate with trust below `theta` is replaced by
the trusted candidate with the lowest level p95. If no candidate reaches the
threshold, the shield chooses the lowest-p95 candidate overall and raises a
stage-specific alert. Other stages are left unchanged.

The default threshold is `theta = 0.5`; the hysteresis hold is `M = 3`
subsequent distinct windows. When an untrusted action triggers a fallback, the
shield continues using the fallback policy for those next three windows,
even if the incoming policy action becomes trusted. An explicit `window_id`
prevents multiple calls within the same window from consuming the hold. If no
window-id provider is configured on `SafetyShieldWrapper`, each call is treated
as the next decision window.

The wrapper defaults to reading trust from the 15 appended trust features
(the expected 98-feature observation: 83 base features plus 15 trust scores).
For other observation layouts, pass a `trust_provider` callback returning a
`(stages, candidates)` array. The shield reads normalized level p95 from the
environment observation's stage/candidate metric blocks.

## Example

```python
from stable_baselines3 import PPO
from defense import SafetyShield, SafetyShieldWrapper
from agents.train_d2 import TrustObservationWrapper
from src.ml_model.data_api_env import DataApiRoutingEnv

policy = PPO.load("results/d2_preliminary/PPO-trust/seed_0/model.zip")
env = SafetyShieldWrapper(
    TrustObservationWrapper(DataApiRoutingEnv(split="val")),
    SafetyShield(threshold=0.5, hold_windows=3),
)
observation, info = env.reset()
action, _ = policy.predict(observation, deterministic=True)
observation, reward, terminated, truncated, info = env.step(action)
print(info["shield"])
```

`info["shield"]` reports overridden stages, fallback stages, alerts, and the
action passed to the underlying environment.

## Validation

`tests/test_shield.py` covers:

- replacing an untrusted choice with the lowest-p95 trusted candidate,
- alerting and choosing the best available candidate when a stage has no
  trusted candidates,
- holding fallback across three subsequent distinct windows and then releasing,
- input validation,
- wrapper-level action override and runtime reporting.

Focused validation command:

```powershell
.\RL_env\Scripts\python.exe -m pytest tests\test_shield.py -q
```

Result: **5 passed**. A separate one-step integration check loaded the saved
`PPO-trust` seed-0 model, wrapped its 98-feature validation environment with
`SafetyShieldWrapper`, and verified that the shield returned a valid
five-stage action and populated the runtime shield information.

## Scope and limitations

This implements the D3 shield rule and hysteresis behavior from the project
guide. It consumes trust scores from an upstream trust layer; it does not
implement or calibrate D1 trust scoring. The current implementation also does
not evaluate shield recovery across attack suites or tune `theta` on clean
validation data.
