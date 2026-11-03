import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from defense.shield import SafetyShield, SafetyShieldWrapper


def test_shield_replaces_untrusted_choice_with_fastest_trusted_candidate():
    shield = SafetyShield(threshold=0.5)
    result = shield.apply(
        action=[0, 1],
        trust_scores=np.array([[0.2, 0.9, 0.8], [0.9, 0.8, 0.7]]),
        level_p95=np.array([[5.0, 4.0, 2.0], [3.0, 2.0, 1.0]]),
        window_id=10,
    )

    np.testing.assert_array_equal(result.action, [2, 1])
    assert result.overridden_stages == (0,)
    assert result.alerts == ()


def test_shield_uses_best_available_and_alerts_when_no_candidate_trusted():
    shield = SafetyShield(threshold=0.5)
    result = shield.apply(
        action=[1],
        trust_scores=np.array([[0.1, 0.2, 0.3]]),
        level_p95=np.array([[3.0, 1.0, 2.0]]),
        window_id=0,
    )

    np.testing.assert_array_equal(result.action, [1])
    assert result.alerts == (0,)
    assert result.overridden_stages == (0,)


def test_shield_hysteresis_holds_fallback_for_three_following_windows():
    shield = SafetyShield(threshold=0.5, hold_windows=3)
    trust = np.array([[0.1, 0.9, 0.8]])
    p95 = np.array([[0.1, 0.8, 0.4]])

    triggered = shield.apply([0], trust, p95, window_id=10)
    assert triggered.action.tolist() == [2]

    for window_id in (11, 12, 13):
        held = shield.apply([0], trust, p95, window_id=window_id)
        assert held.action.tolist() == [2]
        assert held.fallback_stages == (0,)

    released = shield.apply([1], trust, p95, window_id=14)
    assert released.action.tolist() == [1]
    assert released.fallback_stages == ()


def test_shield_validates_input_shapes_and_threshold():
    with pytest.raises(ValueError, match="threshold"):
        SafetyShield(threshold=1.1)

    shield = SafetyShield()
    with pytest.raises(ValueError, match="matching"):
        shield.apply([0], np.ones((1, 2)), np.ones((2, 1)))


class TinyRoutingEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.MultiDiscrete([3, 3])
        self.observation_space = spaces.Box(0.0, 1.0, shape=(36,), dtype=np.float32)
        self.last_action = None
        self._observation = self._make_observation()

    def _make_observation(self):
        obs = np.zeros(36, dtype=np.float32)
        obs[:30] = np.tile(np.array([0.1, 0.1, 0.1, 0.8, 0.0], dtype=np.float32), 6)
        obs[:15].reshape(3, 5)[:, 1] = [0.1, 0.8, 0.4]
        obs[15:30].reshape(3, 5)[:, 1] = [0.2, 0.7, 0.5]
        obs[30:] = [0.1, 0.9, 0.8, 0.9, 0.9, 0.9]
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._observation = self._make_observation()
        return self._observation.copy(), {}

    def step(self, action):
        self.last_action = np.asarray(action).copy()
        return self._observation.copy(), 1.0, False, False, {}


def test_shield_wrapper_overrides_policy_action_and_reports_decision():
    env = TinyRoutingEnv()
    shielded_env = SafetyShieldWrapper(env, SafetyShield(threshold=0.5, hold_windows=0))
    shielded_env.reset()

    _, _, _, _, info = shielded_env.step(np.array([0, 2]))

    np.testing.assert_array_equal(env.last_action, [2, 2])
    assert info["shield"]["overridden_stages"] == [0]
    assert info["shield"]["alerts"] == []
