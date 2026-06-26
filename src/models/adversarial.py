import numpy as np

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

        self.A_dim = (K, T)
        self.A = np.ones(self.A_dim)

        self.X_shape = (self.K, self.T)        # Shape of X matrix
        self.X = np.zeros(self.X_shape)      # Initialise X reward matrix with zeroes

    
    def play(self, t):
        arms = np.arange(self.K)
        active = arms[self.A[:, t]==1]
        return active, self.X[:, t]     # Returns best arm and K x 1 reward vector at time t
    
    def gen_bern(self, rng, p):
        assert isinstance(rng, np.random.Generator), f"expected rng argument to be a numpy random number generator."
        assert len(p) == self.K, f"expected number of parameters {len(p)} same as arms {self.K}."
        
        for t in range(self.T):
            for i in range(self.K):
                self.X[i, t] = rng.binomial(1, p[i])
        
        row_sum = self.X.sum(axis=1)    # K vector of summed values over T
        self.best = np.argmax(row_sum)   # best arm (arg max)

    def genact_bern(self, pa):
        assert len(pa) == self.K, f"expected shape to be K got {len(pa)} instead."

        pa = np.array(pa).reshape(-1, 1)
        self.A = np.random.binomial(1, pa, size=(self.K, self.T))

    # worst case oblivious
    def gen_cycle(self, rng):
        X_shape = (self.K, self.T)
        self.X = np.zeros(X_shape)
        for t in range(self.T):
            self.X[t % self.K, t] = 1
        
        row_sum = self.X.sum(axis=1)    # K vector of summed values over T
        self.best = np.argmax(row_sum)   # best arm (arg max)
    
    def reset(self):
        self.A = np.ones(self.A_dim)
        self.X = np.zeros(self.X_shape)




