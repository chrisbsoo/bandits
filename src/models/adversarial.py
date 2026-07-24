import numpy as np
import warnings

# IN: K, T
# FUNCTIONS OUT:
## GENERATE: K X T MATRIX
## PLAY: K X 1 MATRIX (TIME T)
## INFO: K, T, BEST ARM

class bernoulli_iid_model:
    def __init__(self, K, T):
        assert K >= 0, f"can't have negative arms."
        assert T >= 0, f"can't have negative horizon."
        
        self.new_horizon(K, T)

    def new_horizon(self, K, T):
        self.K = K      # Number of Arms
        self.T = T      # Time Horizon
        self.I = None   # True if it's intermediate setting

        self.A_dim = (K, T)
        self.X_shape = (self.K, self.T)        # Shape of X matrix

        self.reset()
    
    def algo_reset(self):
        if self.I:
            self.smooth_gaussian(self.sigma)
        pass

    def reset(self):
        self.A = np.ones(self.A_dim)
        self.X = np.zeros(self.X_shape)
    
    def play(self, t):
        if np.all(self.X == 0):
            warnings.warn("Reward Matrix is trivially a zero matrix. ")
        arms = np.arange(self.K)
        active = arms[self.A[:, t]==1]
        return active, self.X[:, t]     # Returns best arm and K x 1 reward vector at time t

    # ---- generators ----
    
    def gen_bern(self, rng, p):
        assert isinstance(rng, np.random.Generator), f"expected rng argument to be a numpy random number generator."
        assert len(p) == self.K, f"expected number of parameters {len(p)} same as arms {self.K}."
        
        for t in range(self.T):
            for i in range(self.K):
                self.X[i, t] = rng.binomial(1, p[i])
        
        row_sum = self.X.sum(axis=1)    # K vector of summed values over T
        self.best = np.argmax(row_sum)   # best arm (arg max)
    
    def smooth_gaussian(self, sigma):
        assert len(sigma) == self.K, f"expected variances to have shape {self.K}, got {len(sigma)} instead."
        std = np.array(sigma)[:, None]
        if self.I is None:
            self.sigma = sigma
            self.Xt = self.X.copy()
        self.X = np.random.normal(self.Xt, std)
        self.I = True

    def genact_bern(self, pa, min_awake=1, min_sleep=1):
        assert len(pa) == self.K
        pa = np.array(pa)
        self.A = np.zeros((self.K, self.T))
        
        for i in range(self.K):
            state = np.random.binomial(1, pa[i])  # initial state
            count = 0  # how long in current state
            
            for t in range(self.T):
                self.A[i, t] = state
                count += 1
                
                min_hold = min_awake if state == 1 else min_sleep
                
                if count >= min_hold:
                    # eligible to switch
                    if np.random.binomial(1, pa[i] if state == 0 else 1-pa[i]):
                        state = 1 - state  # flip
                        count = 0

    # worst case oblivious
    def gen_cycle(self, rng):
        X_shape = (self.K, self.T)
        self.X = np.zeros(X_shape)
        for t in range(self.T):
            self.X[t % self.K, t] = 1
        
        row_sum = self.X.sum(axis=1)    # K vector of summed values over T
        self.best = np.argmax(row_sum)   # best arm (arg max)




