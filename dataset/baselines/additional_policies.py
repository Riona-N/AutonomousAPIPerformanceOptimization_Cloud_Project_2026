import random
import numpy as np

class RandomRouting:
    """Choose a candidate randomly."""

    def __init__(self, seed=42):
        self.rng = random.Random(seed)

    def choose(self, candidates):
        return self.rng.choice(list(candidates))


class LeastLoaded:
    """
    Choose the candidate that has received the fewest
    routing assignments so far.
    """

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.counts = {}

    def choose(self, candidates):
        candidates = list(candidates)

        for candidate in candidates:
            self.counts.setdefault(candidate, 0)

        minimum = min(self.counts[c] for c in candidates)

        least_loaded = [
            c for c in candidates
            if self.counts[c] == minimum
        ]

        return self.rng.choice(least_loaded)

    def update(self, candidate):
        self.counts[candidate] = self.counts.get(candidate, 0) + 1


class PowerOfTwoChoices:
    """
    Randomly sample two candidates and choose the one
    with fewer routing assignments.
    """

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.counts = {}

    def choose(self, candidates):
        candidates = list(candidates)

        for candidate in candidates:
            self.counts.setdefault(candidate, 0)

        # If only one candidate is available, choose it.
        if len(candidates) == 1:
            return candidates[0]

        # Randomly sample two different candidates.
        sampled = self.rng.sample(candidates, 2)

        first, second = sampled

        # Choose the less-used candidate.
        if self.counts[first] < self.counts[second]:
            return first

        if self.counts[second] < self.counts[first]:
            return second

        # Break ties randomly.
        return self.rng.choice(sampled)

    def update(self, candidate):
        self.counts[candidate] = self.counts.get(candidate, 0) + 1

class LinUCB:
    """
    Linear Upper Confidence Bound contextual bandit.

    Candidate features must be calculated from training data only.
    """

    def __init__(self, candidate_features, alpha=1.0):
        self.features = {
            int(candidate): np.asarray(features, dtype=float)
            for candidate, features in candidate_features.items()
        }

        self.alpha = alpha
        self.dimension = len(next(iter(self.features.values())))

        self.A = np.eye(self.dimension)
        self.b = np.zeros(self.dimension)

    def choose(self, candidates):
        scores = {}

        theta = np.linalg.solve(self.A, self.b)
        inverse_A = np.linalg.inv(self.A)

        for candidate in candidates:
            x = self.features[int(candidate)]

            predicted_reward = float(x @ theta)
            uncertainty = float(np.sqrt(x @ inverse_A @ x))

            scores[int(candidate)] = (
                predicted_reward + self.alpha * uncertainty
            )

        return max(scores, key=scores.get)

    def update(self, candidate, reward):
        x = self.features[int(candidate)]

        self.A += np.outer(x, x)
        self.b += reward * x
