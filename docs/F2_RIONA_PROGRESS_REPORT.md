# F2 Progress Report: Window-ID Alignment (Riona)

## Purpose

This report covers the F2 task assigned to Riona from the project issue tracker:

> "Window IDs may not align with the latency model's index. Build the mapping once from the data, save it, and test that `latency_model.sample(..., window_ids)` never falls back silently."

The issue is important because the runtime route selection depends on a transaction's `window_id` and on the latency model's internal index. If the two are not aligned, the system may look up the wrong latency bucket, silently produce an incorrect sample, or fail later in a way that is hard to diagnose.

## What is happening

The environment reads two related data artifacts:

- `slot_to_window.parquet`: maps each transaction slot to a `window_id`
- `stage_latency_windows.parquet`: stores the latency profile keyed by `window_id`

The design problem is that a transaction slot has a semantic window value, but the latency model stores rows in a sorted index built from `window_id`. If a slot maps to a window that is missing, malformed, or simply not present in the latency model, the code must not continue with a guessed or silently shifted value.

In other words, the bug pattern is:

1. a slot exists in the transaction table,
2. the slot maps to a `window_id`,
3. the latency model expects a valid window that is in its dataset,
4. if the mapping does not match, the code should stop immediately with a clear error instead of silently sampling the wrong bucket.

## Current implementation behavior

The current code guards this in the environment layer before latency sampling is attempted:

- `DataApiRoutingEnv.__init__()` loads the slot-to-window table and validates that the slot keys are non-negative integers without duplicates.
- `self._valid_window_ids = {int(window) for window in self.latency_model.windows}` is created from the actual latency model windows.
- `_window_for_slot(slot)` checks whether the slot mapping exists, is not null, is an integer-convertible value, and is contained in the valid latency-model window set.
- If any condition fails, the environment raises a `ValueError` naming the offending slot.
- This prevents the model from silently falling back to an index that is not valid for the latency data.

The observation path also uses the same validated window mapping logic when building features for the current transaction and the previous slot, so the environment stays aligned with the data-backed latency model rather than guessing by position.

## Why this matters for the professor review

This is not just a data-cleanliness issue; it is a correctness issue in the RL loop:

- The agent receives observations built from a valid window context.
- Latency sampling uses a valid window and candidate pair.
- Bad mappings are caught early and reported explicitly.
- The training pipeline is protected against silent misalignment between observed windows and the latency model's internal indexing.

Without this guard, an invalid window map could produce incorrect rewards and make the agent learn a flawed policy under the guise of valid data.

## Verification

The repository includes focused tests in `tests/test_data_api_env.py` covering the F2 issue:

1. valid first and last slots are accepted,
2. `reset()` rejects a missing slot mapping,
3. `step()` rejects a missing slot mapping,
4. `reset()` rejects a slot mapped to a window absent from the latency model.

The test command used is:

```powershell
pytest tests\test_data_api_env.py -q
```

Current project status: 4 tests passed on 2026-10-21.

## Summary

This F2 task relates to making the map between transaction slots and latency windows explicit, validated, and fail-fast. The current implementation addresses the issue by checking the mapping at load time and again when the environment reads a slot, ensuring the latency model is never silently accessed with an invalid or stale window index.
