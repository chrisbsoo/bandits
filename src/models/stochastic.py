import numpy as np

# IN: K, T
# FUNCTIONS OUT:
## GENERATE: K X T GAUSSIAN MATRIX
## PLAY: K X 1 MATRIX (TIME T)
## INFO: K, T, BEST ARM

class bernoulli_model:
    def __init__(self, K, T):
        self.K = K
        self.T = T

    def gen_bern(self, p):
        assert len(p) == self.K, f"Expected shape to be K got {len(p)} instead."

        self.p = np.array(p).reshape(-1, 1)

        self.X = np.random.binomial(1, self.p, size=(self.K, self.T))
        best_idx = np.argmax(self.p)
        self.best = best_idx
    
    def play(self, t):
        return self.best, self.X[:, t]
    
    def reset(self):
        self.gen_bern(self.p)

        




