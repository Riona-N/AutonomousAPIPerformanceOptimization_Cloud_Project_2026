import random


class EpsilonGreedyBandit:

    def __init__(self, n_candidates=3, epsilon=0.1, seed=42):

        self.n_candidates = n_candidates
        self.epsilon = epsilon

        self.values = [0.0] * n_candidates
        self.counts = [0] * n_candidates

        self.random = random.Random(seed)

    def choose(self):
        """
        Choose a candidate.

        With probability epsilon:
            explore a random candidate.

        Otherwise:
            exploit the candidate with the highest
            estimated reward.
        """

        if self.random.random() < self.epsilon:
            return self.random.randrange(self.n_candidates)

        return max(
            range(self.n_candidates),
            key=lambda i: self.values[i]
        )

    def update(self, candidate, reward):
        """
        Update the estimated reward of the selected candidate.
        """

        self.counts[candidate] += 1

        count = self.counts[candidate]

        old_value = self.values[candidate]

        self.values[candidate] = (
            old_value
            + (reward - old_value) / count
        )

    def get_values(self):
        """
        Return the current estimated reward
        for each candidate.
        """

        return self.values.copy()

    def get_counts(self):
        """
        Return how many times each candidate
        has been selected.
        """

        return self.counts.copy()