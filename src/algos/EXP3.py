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
    
    def observe(self, active, reward):
        r = reward / self.p[self.i]
        term = (self.gamma * r) / len(active)
        self.W[self.i] = self.W[self.i] * np.exp(term)
    
    def step(self, active):
        t1 = (1-self.gamma) * (self.W[active] / np.sum(self.W))
        t2 = self.gamma / len(active)
        self.p = t1 + t2

        self.p = self.p / self.p.sum()  # renormalisation for float-precision issues.

        assert np.isclose(sum(self.p), 1), f"Expected probabilities to sum up to 1. Got {sum(self.p)} instead."

        self.i = np.random.choice(active, p=self.p)
        return self.i
    
    def reset(self):
        self.W = np.ones(self.K)
        self.p = np.ones(self.K) / self.K
        self.i = None
    
    def eval(self, n_runs=1, ind=1):
        all_regrets = []
        all_regrets_wk = []

        from src.utils import SPINNER
        from src.utils import plot
        
        for run in range(n_runs):
            frame = SPINNER[run % len(SPINNER)]
            
            self.reset()
            regrets = []
            regrets_wk = []
            for t in range(self.model.T):
                print(f'\r{frame} Running... | Episode {run+1}/{n_runs} | Epoch {t+1}/{self.model.T}', end='', flush=True)
                active, rewards = self.model.play(t)
                choice = self.step(active)
                self.observe(active, rewards[choice])
                regrets.append(rewards[self.model.best] - rewards[choice]) # strong regret
                regrets_wk.append(np.max(rewards[active]) - rewards[choice]) # weak regret

            all_regrets.append(np.cumsum(regrets))
            all_regrets_wk.append(np.cumsum(regrets_wk))
        
        plot(n_runs, self.model, all_regrets, all_regrets_wk, ind)






        
    





    


