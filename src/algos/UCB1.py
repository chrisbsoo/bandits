import numpy as np

class simple_ucb1:
    def __init__(self, model):
        self.K = model.K
        self.T = model.T
        self.N = np.zeros(self.K)
        self.p = np.zeros(self.K)
        self.model = model

    def observe(self, reward):
        self.N[self.i] += 1
        self.p[self.i] += (1 / self.N[self.i]) * (reward - self.p[self.i])

    def step(self):
        self.t += 1
        term = (2 * np.log(self.t)) / self.N
        self.UCB = self.p + np.sqrt(term)
        self.i = np.argmax(self.UCB)
        return self.i
    
    def reset(self):
        self.N = np.zeros(self.K)
        self.p = np.zeros(self.K)
        self.i = None
        self.T = 0
        self.t = 0