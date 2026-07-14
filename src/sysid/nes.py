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
        n = len(results)
        order = np.argsort([fitness for _, fitness in results])
        utilities = np.empty(n)
        utilities[order] = np.arange(n) / n - 0.5

        grad = np.zeros_like(self.mean)
        for utility, (params, _) in zip(utilities, results):
            grad += utility * (np.asarray(params) - self.mean)
        grad /= n

        self.mean -= self.alpha * grad