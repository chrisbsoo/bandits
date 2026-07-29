import numpy as np
import numba as nb

# [1, 0] K times

@nb.njit
def make_A_optim(X, ps, pers, K, T):
    timer = np.zeros(K, dtype=np.int64)
    mask = np.zeros((K, T), dtype=np.int64)

    # properly random init at t=0, respecting ps
    while True:
        awakes = np.empty(K, dtype=np.int64)
        for i in range(K):
            awakes[i] = 1 if np.random.random() < ps[i] else 0
        if awakes.any():
            break
    mask[:, 0] = awakes
    for i in range(K):
        timer[i] = pers - 1 if awakes[i] == 1 else 0

    for t in range(1, T):
        prev = mask[:, t-1].copy()
        curr = prev.copy()
        forced = (prev == 1) & (timer > 0)
        timer[forced] -= 1
        redraw = ~forced
        if redraw.any():
            while True:
                awakes = np.empty(K, dtype=np.int64)
                for i in range(K):
                    awakes[i] = 1 if np.random.random() < ps[i] else 0
                if forced.any() or awakes[redraw].any():   # only need global non-emptiness
                    break
            curr[redraw] = awakes[redraw]
            newly_awake = redraw & (prev == 0) & (curr == 1)
            timer[newly_awake] = pers - 1        # <-- fixed off-by-one
        mask[:, t] = curr

    X = X.astype(np.float64)
    for i in range(K):
        for j in range(T):
            if mask[i, j] == 0:
                X[i, j] = np.nan
    return X

class model:
    def __init__(self, K, T):
        self.K = K
        self.T = T

    def make_X(self):
        psx = np.asarray(self.psx)

        self.X = np.random.binomial(
            1,
            psx[:, None],
            size=(self.K, self.T)
        ).astype(np.float64)

        self.best = np.max(psx)


    def make_A(self):
        ps = np.asarray(self.ps, dtype=np.float64)
        self.X = make_A_optim(
            self.X,
            ps,
            self.pers,
            self.K,
            self.T
        )
        

