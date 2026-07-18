import numpy as np
import warnings

# IN: K, T
# FUNCTIONS OUT:
## GENERATE: K X T MATRIX
## PLAY: K X 1 MATRIX (TIME T)
## INFO: K, T, BEST ARM

class intermediate_model:
    def __init__(self, K, T):
        assert K >= 0, f"can't have negative arms."
        assert T >= 0, f"can't have negative horizon."
        
        self.new_horizon(K, T)
    
    def new_horizon(self, K, T):
        self.K = K
        self.T = T
        self.A_dim = (K, T)
        self.A = np.ones(self.A_dim)
    
    def algo_reset(self):
        # resample b_i each Monte Carlo run
        Y_mean = self.Z[:, None] * np.ones((self.K, self.T))
        self.X = np.clip(np.random.normal(Y_mean, self.sigma[:, None]), 0, 1)
    
    def play(self, t):
        arms = np.arange(self.K)
        active = arms[self.A[:, t] == 1]
        return active, self.X[:, t]

    # ---- generators ----

    def gen(self, alpha, beta, sigma):
        self.alpha = np.array(alpha)
        self.beta = np.array(beta)
        self.sigma = np.array(sigma)

        self.b = np.random.binomial(1, 0.5, size=self.K)
        self.Zr = self.alpha + self.beta * self.b      # (K,) fixed per run
        self.Z = 3 * self.sigma + (self.Zr - self.Zr.min()) / (self.Zr.max() - self.Zr.min()) * (1 - 6 * self.sigma)
        self.best = np.argmax(self.Z)
        self.algo_reset()