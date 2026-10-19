import random
import numpy as np
import math
import random


class EpsilonGreedyBandit:
    def __init__(self, n_candidates=3, epsilon=0.1, seed=42):
        self.n_candidates = n_candidates
        self.epsilon = epsilon
        self.values = [0.0] * n_candidates
        self.counts = [0] * n_candidates
        self.random = random.Random(seed)

    def choose(self):
        if self.random.random() < self.epsilon:
            return self.random.randrange(self.n_candidates)

        return max(
            range(self.n_candidates),
            key=lambda i: self.values[i]
        )

    def update(self, candidate, reward):
        self.counts[candidate] += 1
        count = self.counts[candidate]
        old_value = self.values[candidate]

        self.values[candidate] = (
            old_value + (reward - old_value) / count
        )

    def get_values(self):
        return self.values.copy()

    def get_counts(self):
        return self.counts.copy()



class UCB1:
    def __init__(self, n_candidates=3):
        self.n = n_candidates
        self.counts = [0] * n_candidates
        self.values = [0.0] * n_candidates
        self.total = 0

    def choose(self):
        # Explore each candidate once before calculating UCB scores.
        for arm in range(self.n):
            if self.counts[arm] == 0:
                return arm

        total_counts = sum(self.counts)
        ucb_scores = []

        for arm in range(self.n):
            average_reward = self.values[arm]
            exploration_bonus = (
                2 * math.log(total_counts) / self.counts[arm]
            ) ** 0.5

            ucb_scores.append(average_reward + exploration_bonus)

        return max(range(self.n), key=lambda arm: ucb_scores[arm])

    def update(self, candidate, reward):
        self.total += 1
        self.counts[candidate] += 1

        count = self.counts[candidate]
        old_value = self.values[candidate]

        self.values[candidate] = (
            old_value + (reward - old_value) / count
        )

    def get_values(self):
        return self.values.copy()

    def get_counts(self):
        return self.counts.copy()


class ThompsonSampling:
    def __init__(self, n_candidates=3, seed=42):
        self.n = n_candidates
        self.counts = [0] * n_candidates
        self.values = [0.0] * n_candidates
        self.rng = np.random.default_rng(seed)

    def choose(self):
        # Sample a possible reward for each candidate
        scores = []

        for i in range(self.n):
            uncertainty = (
                1 / (self.counts[i] + 1)
            ) ** 0.5

            sampled_reward = self.rng.normal(
                self.values[i],
                uncertainty
            )

            scores.append(sampled_reward)

        return int(np.argmax(scores))

    def update(self, candidate, reward):
        self.counts[candidate] += 1

        count = self.counts[candidate]
        old_value = self.values[candidate]

        self.values[candidate] = (
            old_value + (reward - old_value) / count
        )

    def get_values(self):
        return self.values.copy()

    def get_counts(self):
        return self.counts.copy()