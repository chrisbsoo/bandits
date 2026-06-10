import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

SPINNER = ['/', '-', '\\', '|']

class EXP3_policy:
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

    def eval_multi(self, n_runs):
        all_regrets = []
        
        for run in range(n_runs):
            frame = SPINNER[run % len(SPINNER)]
            
            self.reset()
            regrets = []
            for t in range(self.T):
                print(f'\r{frame} Running... | Episode {run+1}/{n_runs} | Epoch {t+1}/{self.T}', end='', flush=True)
                choice = self.step()
                best, rewards = self.model.play(t)
                self.observe(rewards[choice])
                regrets.append(rewards[best] - rewards[choice])
            
            all_regrets.append(np.cumsum(regrets))
        
        all_regrets = np.array(all_regrets)   # (n_runs, T)
        mean = all_regrets.mean(axis=0)       # (T,)
        std = all_regrets.std(axis=0)         # (T,)
        
        t = np.arange(1, self.T+1)
        bound = 2 * np.sqrt((np.e - 1) * t * self.K * np.log(self.K))
        
        plt.plot(t, mean, label='EXP3 Mean Regret', zorder=4)
        plt.fill_between(t, mean - std, mean + std, alpha=0.5, label='±1 std', zorder=3)
        plt.fill_between(t, mean - 2*std, mean + 2*std, alpha=0.5, label='±2 std', zorder=2)
        plt.fill_between(t, mean - 3*std, mean + 3*std, alpha=0.5, label='±3 std', zorder=1)
        plt.plot(t, bound, label='Theoretical Bound', linestyle='--', color='red', zorder=4)
        plt.xlabel('Time')
        plt.ylabel('Cumulative Regret')
        plt.legend()
        plt.show()
        
        return mean, std
    




        
    





    


