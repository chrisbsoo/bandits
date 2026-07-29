import numba as nb
import numpy as np

@nb.njit
def exp3_loop(X, ps, best, gamma, weak):
    K, T = X.shape
    w = np.ones(K)
    p = np.zeros(K)
    reg = np.zeros(T)

    for t in range(T):
        active_sum = 0.0
        n_active = 0
        best_active = -np.inf

        for i in range(K):
            if not np.isnan(X[i, t]):
                active_sum += w[i]
                n_active += 1
                if ps[i] > best_active:
                    best_active = ps[i]

        # build sampling distribution over active arms only
        for i in range(K):
            if not np.isnan(X[i, t]):
                p[i] = (1.0 - gamma) * (w[i] / active_sum) + gamma / n_active
            else:
                p[i] = 0.0

        # sample an arm according to p
        u = np.random.random()
        cum = 0.0
        step = -1
        for i in range(K):
            cum += p[i]
            if u <= cum:
                step = i
                break
        if step == -1:  # numerical safety net
            for i in range(K):
                if p[i] > 0.0:
                    step = i

        observed = X[step, t]

        # importance-weighted reward estimate, update only chosen arm's weight
        est_reward = observed / p[step]
        w[step] *= np.exp(gamma * est_reward / K)

        if weak:
            reg[t] = best_active - ps[step]
        else:
            reg[t] = best - ps[step]

    return np.cumsum(reg)


class exp3_policy:

    def __init__(self, model):
        self.model = model

    def play(self, gamma=0.1, weak=False):
        return exp3_loop(
            self.model.X,
            self.model.psx,
            self.model.best,
            gamma,
            weak
        )