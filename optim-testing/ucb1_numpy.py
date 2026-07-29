import numpy as np
import numba as nb

class ucb1_policy:

    def __init__(self, model):
        self.model = model
        self.K = model.K
        self.T = model.T
        self.reset()

    def play(self, weak=False):
        self.reset()

        reg = np.zeros(self.T)

        for t in range(1, self.T+1):
            idxs = np.flatnonzero(~np.isnan(self.model.X[:, t-1]))

            step = idxs[np.argmax(self.ucb[idxs])]
            observed = self.model.X[step, t-1]

            self.n[step] += 1

            self.mu[step] += (observed - self.mu[step]) / self.n[step]

            actives = self.n > 0
            self.ucb[actives] = self.mu[actives] + np.sqrt(2 * np.log(t) / self.n[actives])

            if weak:
                reg[t-1] = np.max(self.model.X[idxs, t-1]) - observed
            else:
                reg[t-1] = self.model.best - observed

        return np.cumsum(reg)

    def reset(self):
        self.mu, self.n = np.zeros(self.K), np.zeros(self.K)
        self.ucb = np.full(self.K, np.inf)