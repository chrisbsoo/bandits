import numpy as np

class simple_ucb1:
    def __init__(self, model):
        self.K = model.K
        self.T = model.T
        self.N = np.zeros(self.K)
        self.p = np.zeros(self.K)
        self.UCB = np.full(self.K, np.inf)
        self.model = model

    def observe(self, reward):
        self.N[self.i] += 1
        self.p[self.i] += (1 / self.N[self.i]) * (reward - self.p[self.i])

    def step(self, active):
        self.t += 1
        term = (2 * np.log(self.t)) / self.N[active]
        self.UCB[active] = self.p[active] + np.sqrt(term)
        self.i = np.argmax(self.UCB[active])
        return self.i
    
    def reset(self):
        self.N = np.zeros(self.K)
        self.p = np.zeros(self.K)
        self.i = None
        self.T = 0
        self.t = 0
    
    def eval(self, n_runs=1, ind=1):
        from src.utils import SPINNER
        from src.utils import plot
        all_regrets = []
        all_regrets_wk = []
        
        for run in range(n_runs):

            frame = SPINNER[run % len(SPINNER)]
            self.reset()
            self.model.reset()
            regrets = []
            regrets_wk = []

            for t in range(self.model.T):
                print(f'\r{frame} Running... | Episode {run+1}/{n_runs} | Epoch {t+1}/{self.model.T}', end='', flush=True)
                active, rewards = self.model.play(t)
                if not active:
                    continue

                choice = self.step(active)
                self.observe(rewards[choice])
                
                regrets.append(rewards[self.model.best] - rewards[choice])
                regrets_wk.append(np.max(rewards[active]) - rewards[choice]) # weak regret
            
            all_regrets.append(np.cumsum(regrets))
            all_regrets_wk.append(np.cumsum(regrets_wk))
        
        plot(n_runs, self.model, all_regrets, all_regrets_wk, ind)