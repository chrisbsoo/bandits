import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from src.models.stochastic import gaussian_model

SPINNER = ['/', '-', '\\', '|']

class UCB1_policy:
    def __init__(self, model):
        self.K = model.K
        self.T = model.T
        self.N = np.zeros(self.K)
        self.mu = np.zeros(self.K)
        self.model = model

    def observe(self, reward):
        self.N[self.i] += 1
        self.mu[self.i] += (1 / self.N[self.i]) * (reward - self.mu[self.i])

    def step(self):
        term = (2 * np.log(self.t)) / self.N
        self.UCB = self.mu + np.sqrt(term)
        chosen = np.argmax(self.UCB)
        self.i = chosen
        return chosen
    
    def reset(self, run):
        self.N = np.zeros(self.K)
        self.mu = np.zeros(self.K)
        self.i = None
        self.t = 0
        # regenerate model with new data
        self.model = gaussian_model.from_gauss(
            self.K, self.T,
            self.model.mu.flatten(),
            self.model.sigma.flatten()
        )

    def eval_multi(self, n_runs):
        all_regrets = []
        
        for run in range(n_runs):
            frame = SPINNER[run % len(SPINNER)]
            self.reset(run)
            regrets = []

            for t in range(self.K):
                print(f'\r{frame} Running... | Episode {run+1}/{n_runs} | Epoch {t+1}/{self.T}', end='', flush=True)
                self.t = t
                best, rewards = self.model.play(t)
                self.N[t] += 1
                self.mu[t] = rewards[t]
                regrets.append(rewards[best] - rewards[t])

            for t in range(self.K, self.T):
                print(f'\r{frame} Running... | Episode {run+1}/{n_runs} | Epoch {t+1}/{self.T}', end='', flush=True)
                self.t = t
                choice = self.step()
                best, rewards = self.model.play(t)
                self.observe(rewards[choice])
                regrets.append(rewards[best] - rewards[choice])
            
            all_regrets.append(np.cumsum(regrets))
        
        all_regrets = np.array(all_regrets)   # (n_runs, T)
        mean = all_regrets.mean(axis=0)
        std = all_regrets.std(axis=0)

        t = np.arange(1, self.T+1)
        deltas = max(self.model.mu) - self.model.mu
        deltas = deltas[deltas > 0]
        bound = np.sum(8 * np.log(t) / deltas[:, None] + (1 + np.pi**2 / 3) * deltas[:, None], axis=0)

        plt.fill_between(t, np.maximum(mean - 3*std, 0), mean + 3*std, alpha=0.1, label='±3 std', zorder=1)
        plt.fill_between(t, np.maximum(mean - 2*std, 0), mean + 2*std, alpha=0.2, label='±2 std', zorder=2)
        plt.fill_between(t, np.maximum(mean - std, 0), mean + std, alpha=0.3, label='±1 std', zorder=3)
        plt.plot(t, mean, label='UCB1 Mean Regret', zorder=4)
        plt.plot(t, bound, label='Theoretical Bound', linestyle='--', color='red', zorder=5)
        plt.xlabel('Time')
        plt.ylabel('Cumulative Regret')
        plt.legend()
        plt.show()

        return mean, std