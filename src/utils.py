import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

SPINNER = ['/', '-', '\\', '|']


def eval_adv(model, algo, n_runs=None):
    all_regrets = []
    
    for run in range(n_runs):
        frame = SPINNER[run % len(SPINNER)]
        
        algo.reset()
        regrets = []
        for t in range(model.T):
            print(f'\r{frame} Running... | Episode {run+1}/{n_runs} | Epoch {t+1}/{model.T}', end='', flush=True)
            choice = algo.step()
            best, rewards = model.play(t)
            algo.observe(rewards[choice])
            regrets.append(rewards[best] - rewards[choice])
        
        all_regrets.append(np.cumsum(regrets))
    
    all_regrets = np.array(all_regrets)   # (n_runs, T)
    mean = all_regrets.mean(axis=0)       # (T,)
    std = all_regrets.std(axis=0)         # (T,)
    
    t = np.arange(1, model.T+1)
    bound = 2 * np.sqrt((np.e - 1) * t * model.K * np.log(model.K))
    
    plt.title(f"Time (X) vs CummReg (Y), {n_runs} iid runs, {model.T} T, {model.K} K")
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
    

def eval_stoch(model, algo, n_runs):
    all_regrets = []
    
    for run in range(n_runs):

        frame = SPINNER[run % len(SPINNER)]
        algo.reset()
        model.reset()
        regrets = []
        for t in range(model.K):
            
            print(f'\r{frame} Running... | Episode {run+1}/{n_runs} | Epoch {t+1}/{model.T}', end='', flush=True)
            best, rewards = model.play(t)
            algo.i = t
            algo.observe(rewards[t])
            regrets.append(rewards[best] - rewards[t])

        for t in range(model.K, model.T):
            
            print(f'\r{frame} Running... | Episode {run+1}/{n_runs} | Epoch {t+1}/{model.T}', end='', flush=True)
            choice = algo.step()
            best, rewards = model.play(t)
            algo.observe(rewards[choice])

            regrets.append(rewards[best] - rewards[choice])
        
        all_regrets.append(np.cumsum(regrets))
    
    all_regrets = np.array(all_regrets)   # (n_runs, T)
    mean = all_regrets.mean(axis=0)       # (T,)
    std = all_regrets.std(axis=0)         # (T,)
    
    t = np.arange(1, model.T+1)
    p_flat = model.p.flatten()
    delta = p_flat[model.best] - np.sort(p_flat)[-2] # smallest gap across suboptimal arms
    bound = model.K * (8 * np.log(t) / delta + 1 + np.pi**2 / 3 * delta)
    
    plt.title(f"Time (X) vs CummReg (Y), {n_runs} iid runs, {model.T} T, {model.K} K")
    plt.plot(t, mean, label='UCB1 Mean Regret', zorder=4)
    plt.fill_between(t, mean - std, mean + std, alpha=0.5, label='±1 std', zorder=3)
    plt.fill_between(t, mean - 2*std, mean + 2*std, alpha=0.5, label='±2 std', zorder=2)
    plt.fill_between(t, mean - 3*std, mean + 3*std, alpha=0.5, label='±3 std', zorder=1)
    plt.plot(t, bound, label='Theoretical Bound', linestyle='--', color='red', zorder=4)
    plt.xlabel('Time')
    plt.ylabel('Log Cumulative Regret')
    plt.yscale('log')
    plt.legend()
    plt.show()
    
    return mean, std

    