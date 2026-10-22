# F3 Progress Report: Post-Termination Stepping (Riona)

## Purpose

This report covers the F3 issue assigned to Riona from the project tracker:

> "Caller can keep stepping after terminated=True. Raise an error if step() is called after termination."

This is a standard Gymnasium-style environment contract issue. Once an episode is marked as terminated, the environment should reject any further calls to `step()` until the caller invokes `reset()`. Continuing to step after termination violates the training loop contract and can create invalid transitions, stale observations, and misleading reward traces.

## What is happening

In the data-backed RL environment, the termination condition is computed in `step()` as:

```python
self._position += 1
terminated = self._position >= min(
    self.episode_length, len(self.transactions) - self._episode_start
)
```

The method returns `terminated` in the step result, but there is no guard at the beginning of the method to stop the caller from making additional calls after the episode is already terminated. This means the following sequence is currently possible:

1. `reset()` starts a new episode.
2. `step()` is called repeatedly until the episode ends.
3. The environment returns `terminated=True`.
4. The caller can still call `step()` again without a reset.
5. The environment may continue processing another transaction or perform another observation update even though the episode should be closed.

That is the bug the issue is flagging.

## Why this is a problem

A terminated episode is a terminal state. After that point:

- no further transitions should be sampled,
- the episode should not advance to the next transaction,
- the environment should not accept additional action inputs,
- the caller must reset before continuing.

If `step()` remains callable after termination, the RL agent or trainer can accidentally generate invalid data or silently continue training from a non-reset environment, which can distort evaluation and learning behavior.

## Current implementation behavior in this project

The environment currently performs the following in `src/ml_model/data_api_env.py`:

- It tracks the current transaction index in `self._position`.
- It computes `terminated` based on the episode length and current position.
- It returns `(observation, reward, terminated, False, info)`.
- There is no explicit check at the start of `step()` such as:

```python
if self._position >= min(self.episode_length, len(self.transactions) - self._episode_start):
    raise RuntimeError("Cannot call step() after termination; call reset() first")
```

This means the environment currently accepts illegitimate post-termination stepping instead of enforcing the expected contract.

## Correct behavior expected for F3

The post-termination contract should be fail-fast:

- if the previous step returned `terminated=True`, the next call to `step()` should raise an error,
- the error should clearly instruct the user to call `reset()` before continuing,
- no new reward, action, or transaction processing should be executed after termination.

This matches standard RL environment behavior and prevents hidden logic errors in the training loop.

## Why this matters for professor review

This is not a cosmetic issue; it is a correctness and training-safety issue. In reinforcement learning, the environment is expected to enforce episode lifecycle rules. If those rules are not enforced, the state machine becomes ambiguous:

- the agent may believe the episode is over while the environment continues,
- the training script may collect invalid transition data,
- reward interpretation can become unreliable,
- evaluation metrics can be silently corrupted.

A robust environment should therefore make it impossible to keep stepping after termination.

## Recommended fix

The environment should add a guard at the top of `step()` to reject calls after the episode has terminated. The guard should check the current position against the episode boundary before reading the next transaction or sampling action outcomes.

A typical implementation would look like this:

```python
if self._position >= min(self.episode_length, len(self.transactions) - self._episode_start):
    raise RuntimeError("Cannot call step() after termination. Call reset() first.")
```

This ensures that the environment is strict, consistent, and safe for RL training.

## Verification approach

A focused test should cover the regression:

1. create environment and call `reset()`;
2. advance the environment until it terminates;
3. assert that the next `step()` raises a `RuntimeError`;
4. confirm that the environment only accepts a new episode after `reset()`.

This would validate the intended behavior and provide evidence that the training loop remains consistent.

## Fix applied and validation status

The F3 fix has been implemented and verified in the codebase.

- The environment now guards against post-termination stepping in `src/ml_model/data_api_env.py`.
- The guard raises:

```python
RuntimeError("Cannot call step() after termination. Call reset() first.")
```

- A regression test was added in `tests/test_data_api_env.py` to ensure the environment rejects a second `step()` call after `terminated=True`.

Validation command:

```powershell
Set-Location "c:\Users\mousu\AutonomousAPIPerformanceOptimization_Cloud_Project_2026"
.\RL_env\Scripts\python.exe -m pytest tests\test_data_api_env.py -q
```

Result: 6 tests passed.

## Summary

F3 is the environment contract fix for post-termination stepping. The issue is that the environment returns `terminated=True` but does not reject additional calls to `step()`. The correct fix is to raise an error and require `reset()`, which aligns the environment with Gymnasium conventions and protects the RL pipeline from invalid continuation after episode end.
