"""Role 3 routing-policy interface tests.
Run: python -m unittest discover -s tests -p 'test_interfaces.py' -v
"""
import math
import unittest

from dataset.baselines.bandit import EpsilonGreedyBandit, ThompsonSampling, UCB1
from dataset.baselines.additional_policies import (
    LinUCB, LeastLoaded, PowerOfTwoChoices, RandomRouting,
)


class PolicyTests(unittest.TestCase):
    def test_random_is_seeded_and_returns_candidate(self):
        a, b = RandomRouting(seed=17), RandomRouting(seed=17)
        candidates = [0, 1, 2]
        xs = [a.choose(candidates) for _ in range(20)]
        ys = [b.choose(candidates) for _ in range(20)]
        self.assertEqual(xs, ys)
        self.assertTrue(all(x in candidates for x in xs))

    def test_least_loaded_updates_counts(self):
        p = LeastLoaded(seed=17)
        x = p.choose([0, 1, 2])
        p.update(x)
        self.assertEqual(p.counts[x], 1)
        self.assertIn(p.choose([0, 1, 2]), [0, 1, 2])

    def test_power_of_two_single_candidate(self):
        p = PowerOfTwoChoices(seed=17)
        self.assertEqual(p.choose([2]), 2)
        p.update(2)
        self.assertEqual(p.counts[2], 1)

    def test_power_of_two_returns_available_candidate(self):
        p = PowerOfTwoChoices(seed=17)
        self.assertTrue(all(p.choose([0, 2]) in [0, 2] for _ in range(20)))

    def test_epsilon_greedy(self):
        p = EpsilonGreedyBandit(n_candidates=3, epsilon=0.1, seed=17)
        arm = p.choose()
        self.assertIn(arm, range(3))
        p.update(arm, 0.8)
        self.assertEqual(sum(p.get_counts()), 1)
        self.assertTrue(all(math.isfinite(float(x)) for x in p.get_values()))
        
    def test_ucb1_explores_all_arms(self):
        p = UCB1(n_candidates=3)
        arms = []
        for _ in range(3):
            arm = p.choose()
            arms.append(arm)
            p.update(arm, 0.5)
        self.assertEqual(set(arms), {0, 1, 2})
        self.assertEqual(sum(p.get_counts()), 3)
        self.assertIn(p.choose(), range(3))

    def test_thompson_sampling(self):
        p = ThompsonSampling(n_candidates=3, seed=17)
        arm = p.choose()
        self.assertIn(arm, range(3))
        p.update(arm, 0.7)
        self.assertEqual(sum(p.get_counts()), 1)
        self.assertTrue(all(math.isfinite(float(x)) for x in p.get_values()))

    def test_linucb_choose_update(self):
        p = LinUCB({0: [1.0, 0.0], 1: [0.0, 1.0], 2: [1.0, 1.0]}, alpha=1.0)
        arm = p.choose([0, 1, 2])
        self.assertIn(arm, [0, 1, 2])
        p.update(arm, 0.5)
        self.assertIn(p.choose([0, 1, 2]), [0, 1, 2])

    def test_linucb_single_candidate(self):
        p = LinUCB({2: [1.0, 0.0]}, alpha=1.0)
        self.assertEqual(p.choose([2]), 2)
        p.update(2, 0.4)


if __name__ == "__main__":
    unittest.main()
