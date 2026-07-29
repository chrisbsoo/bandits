import numba as nb
import numpy as np

@nb.njit
def ucb1_loop(X, ps, best, weak):
    K, T = X.shape
    mu = np.zeros(K)
    n = np.zeros(K)
    ucb = np.full(K, np.inf)
    reg = np.zeros(T)

    for t in range(1, T + 1):
        max_ucb = -np.inf
        best_active = -np.inf
        step = -1

        for i in range(K):
            reward = X[i, t - 1]
            if not np.isnan(reward):
                if ps[i] > best_active:
                    best_active = ps[i]
                if ucb[i] > max_ucb:
                    max_ucb = ucb[i]
                    step = i

        observed = X[step, t - 1]
        n[step] += 1
        mu[step] += (observed - mu[step]) / n[step]

        logt = np.log(t)
        for i in range(K):
            if n[i] > 0:
                ucb[i] = mu[i] + np.sqrt(2.0 * logt / n[i])

        if weak:
            reg[t - 1] = best_active - ps[step]
        else:
            reg[t - 1] = best - ps[step]

    return np.cumsum(reg)


class ucb1_policy:

    def __init__(self, model):
        self.model = model

    def play(self, weak=False):
        return ucb1_loop(
            self.model.X,
            self.model.psx,
            self.model.best,
            weak
        )