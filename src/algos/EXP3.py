import numpy as np

class simple_exp3:
    def __init__(self, model, gamma=None):
        self.K = model.K
        self.T = model.T
        self.model = model
        self.W = np.ones(self.K)

        if gamma is None:
            u = self.K * np.log(self.K)
            d = (np.e - 1) * self.T
            term = np.sqrt(u / d)
            self.gamma = min(1, term)
        else:
            self.gamma = gamma
    
    def observe(self, reward):
        r = reward / self.p[self.i]
        term = (self.gamma * r) / self.K
        self.W[self.i] = self.W[self.i] * np.exp(term)
    
    def step(self):
        t1 = (1-self.gamma) * (self.W / np.sum(self.W))
        t2 = self.gamma / self.K
        self.p = t1 + t2

        self.p = self.p / self.p.sum()  # renormalisation for float-precision issues.

        assert np.isclose(sum(self.p), 1), f"Expected probabilities to sum up to 1. Got {sum(self.p)} instead."

        self.i = np.random.choice(self.K, p=self.p)
        return self.i
    
    def reset(self):
        self.W = np.ones(self.K)
        self.p = np.ones(self.K) / self.K
        self.i = None






        
    





    


