import numpy as np

# IN: K, T
# FUNCTIONS OUT:
## GENERATE: K X T GAUSSIAN MATRIX
## PLAY: K X 1 MATRIX (TIME T)
## INFO: K, T, BEST ARM

class gaussian_model:
    def __init__(self, K, T):
        self.K = K
        self.T = T

    def gen_gauss(self, mu, sigma):
        assert len(mu) == self.K, f"Expected shape to be K got {len(mu)} instead."
        assert len(sigma) == self.K, f"Expected shape to be K got {len(sigma)} instead."

        self.mu = np.array(mu).reshape(-1, 1)
        self.sigma = np.array(sigma).reshape(-1, 1)

        self.X = np.random.normal(self.mu, self.sigma, size=(self.K, self.T))
    
    def play(self, t):
        best = np.argmax(self.mu)
        return best, self.X[:, t]

    @classmethod
    def from_gauss(cls, K, T, mu, sigma):
        model = cls(K, T)
        model.gen_gauss(mu, sigma)
        return model

        




