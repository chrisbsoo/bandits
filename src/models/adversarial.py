import numpy as np

# IN: K, T
# FUNCTIONS OUT:
## GENERATE: K X T MATRIX
## PLAY: K X 1 MATRIX (TIME T)
## INFO: K, T, BEST ARM

class bernoulli_iid_model:
    def __init__(self, K, T):
        self.K = K      # Number of Arms
        self.T = T      # Time Horizon
    
    def play(self, t):
        row_sum = self.X.sum(axis=1)    # K vector of summed values over T
        best = np.argmax(row_sum)   # best arm (arg max)

        return best, self.X[:, t]     # Returns best arm and K x 1 reward vector at time t
    
    def gen_bern(self, rng, p):
        if len(p) != self.K:
            raise ValueError(f"Number of parameters {len(p)} not same as number of arms {self.K}")
        
        X_shape = (self.K, self.T)        # Shape of X matrix
        self.X = np.zeros(X_shape)      # Initialise X reward matrix with zeroes
        
        for t in range(self.T):
            for i in range(self.K):
                self.X[i, t] = rng.binomial(1, p[i])
        
        return self.X


