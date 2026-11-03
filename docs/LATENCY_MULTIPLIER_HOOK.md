# Latency Multiplier Hook

## Purpose

The environment now includes a configurable `latency_multiplier` hook so the sampled stage latencies can be scaled at rollout time without editing the latency model itself.

## Implementation

The setting is defined in `src/config.py` as:

```python
LATENCY_MULTIPLIER = 1.0
```

The environment in `src/ml_model/data_api_env.py` reads this value and exposes an override during initialization:

```python
DataApiRoutingEnv(..., latency_multiplier=0.5)
```

If no override is provided, the environment uses the global config value. The hook is applied to the sampled per-stage latencies before the end-to-end latency and reward are computed.

## Why this matters

This creates a clean, reviewable control point for experiments that want to simulate:

- slower or faster backend behavior,
- degraded latency conditions,
- sensitivity testing for SLA and timeout thresholds,
- and reproducible experiment tuning without altering the source dataset.

## Validation

The regression test in `tests/test_data_api_env.py` verifies that setting `latency_multiplier=0.0` yields zero latency in the returned step information.

This allows the project to tune latency conditions consistently while keeping the underlying trace data fixed.
