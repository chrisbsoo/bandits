import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

SPINNER = ['/', '-', '\\', '|']

def plot(n_runs, model, all_regrets, all_regrets_wk, show=2):
    all_regrets = np.array(all_regrets)   # (n_runs, T)
    all_regrets_wk = np.array(all_regrets_wk)   # (n_runs, T)
    mean_wk = all_regrets_wk.mean(axis=0)   # (T,)
    std_wk = all_regrets_wk.std(axis=0)     # (T,)
    mean = all_regrets.mean(axis=0)       # (T,)
    std = all_regrets.std(axis=0)         # (T,)
    
    t = np.arange(1, model.T+1)
    
    plt.title(f"Time (X) vs CummReg (Y), {n_runs} iid runs, {model.T} T, {model.K} K")
    if show == 1 or show == 2:
        plt.plot(t, mean, label='Strong Mean Regret', zorder=4)
        plt.fill_between(t, mean - std, mean + std, alpha=0.5, label='±1 std', zorder=3)
    if show == 0 or show == 2:
        plt.plot(t, mean_wk, label='Weak Mean Regret', zorder=4)
        plt.fill_between(t, mean_wk - std_wk, mean_wk + std_wk, alpha=0.5, label='±1 std', zorder=3)
    plt.xlabel('Time')
    plt.ylabel('Cumulative Regret')
    plt.legend()
    plt.show()

def eval_adv(model, algo, n_runs=None, ind=2):
    all_regrets = []
    all_regrets_wk = []
    
    for run in range(n_runs):
        frame = SPINNER[run % len(SPINNER)]
        
        algo.reset()
        regrets = []
        regrets_wk = []
        for t in range(model.T):
            print(f'\r{frame} Running... | Episode {run+1}/{n_runs} | Epoch {t+1}/{model.T}', end='', flush=True)
            choice = algo.step()
            rewards = model.play(t)
            algo.observe(rewards[choice])
            regrets.append(rewards[model.best] - rewards[choice]) # strong regret
            regrets_wk.append(np.max(rewards) - rewards[choice]) # weak regret

        all_regrets.append(np.cumsum(regrets))
        all_regrets_wk.append(np.cumsum(regrets_wk))
    
    plot(n_runs, model, all_regrets, all_regrets_wk, ind)
    

def eval_stoch(model, algo, n_runs=None, ind=2):
    all_regrets = []
    all_regrets_wk = []
    
    for run in range(n_runs):

        frame = SPINNER[run % len(SPINNER)]
        algo.reset()
        model.reset()
        regrets = []
        regrets_wk = []
        for t in range(model.K):
            
            print(f'\r{frame} Running... | Episode {run+1}/{n_runs} | Epoch {t+1}/{model.T}', end='', flush=True)
            rewards = model.play(t)
            algo.i = t
            algo.observe(rewards[t])
            regrets.append(rewards[model.best] - rewards[t])
            regrets_wk.append(np.max(rewards) - rewards[t]) # weak regret

        for t in range(model.K, model.T):
            
            print(f'\r{frame} Running... | Episode {run+1}/{n_runs} | Epoch {t+1}/{model.T}', end='', flush=True)
            choice = algo.step()
            rewards = model.play(t)
            algo.observe(rewards[choice])

            regrets.append(rewards[model.best] - rewards[choice])
            regrets_wk.append(np.max(rewards) - rewards[choice]) # weak regret
        
        all_regrets.append(np.cumsum(regrets))
        all_regrets_wk.append(np.cumsum(regrets_wk))
    
    plot(n_runs, model, all_regrets, all_regrets_wk, ind)
    

    