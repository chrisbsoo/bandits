import numpy as np

# IN: K, T
# FUNCTIONS OUT:
## GENERATE: K X T GAUSSIAN MATRIX
## PLAY: K X 1 MATRIX (TIME T)
## INFO: K, T, BEST ARM

class bernoulli_model:
    def __init__(self, K, T):
        assert K >= 0, f"can't have negative arms."
        assert T >= 0, f"can't have negative horizon."
        self.new_horizon(K, T)
    
    def new_horizon(self, K, T):
        self.K = K
        self.T = T
        self.A_dim = (K, T)
        self.A = np.ones(self.A_dim)

    def gen_bern(self, p):
        assert len(p) == self.K, f"expected shape to be K got {len(p)} instead."

        self.p = np.array(p).reshape(-1, 1)
        self.X = np.random.binomial(1, self.p, size=(self.K, self.T))
        best_idx = np.argmax(self.p)
        self.best = best_idx
    
    def genact_bern(self, pa):
        assert len(pa) == self.K, f"expected shape to be K got {len(pa)} instead."

        pa = np.array(pa).reshape(-1, 1)
        self.A = np.random.binomial(1, pa, size=(self.K, self.T))
    
    def play(self, t):
        arms = np.arange(self.K)
        active = arms[self.A[:, t]==1]
        return active, self.X[:, t]     # Returns best arm and K x 1 reward vector at time t
    
    def algo_reset(self):
        self.gen_bern(self.p)
    
    def reset(self):
        self.gen_bern(self.p)
        self.A = np.ones(self.A_dim)


        




