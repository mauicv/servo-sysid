import numpy as np


class NES:
    def __init__(self, mean, sigma, population_size, alpha):
        self.mean = np.asarray(mean, dtype=float)
        self.sigma = sigma
        self.population_size = population_size
        self.alpha = alpha

    def ask(self):
        return self.mean + self.sigma * np.random.randn(len(self.mean))

    def tell(self, results):
        # results is a list of tuples (params, fitness), fitness is a LOSS to
        # minimize. We shape fitness by RANK rather than using raw loss values:
        # sort ascending by loss and spread utilities evenly over [-0.5, 0.5],
        # so the lowest-loss sample gets -0.5 and the highest gets ~+0.5. This
        # makes the step invariant to the scale/outliers of the loss and removes
        # the need for a separate baseline (the utilities are already centred).
        #
        # Natural-gradient NES mean update (the 1/sigma**2 cancels for a fixed
        # isotropic Gaussian): grad = (1/N) * sum utility * (params - mean).
        # We DESCEND: a low-loss sample has a negative utility, so subtracting
        # the gradient pulls the mean toward it.
        n = len(results)
        order = np.argsort([fitness for _, fitness in results])  # ascending loss
        utilities = np.empty(n)
        utilities[order] = np.arange(n) / n - 0.5

        grad = np.zeros_like(self.mean)
        for utility, (params, _) in zip(utilities, results):
            grad += utility * (np.asarray(params) - self.mean)
        grad /= n

        self.mean -= self.alpha * grad