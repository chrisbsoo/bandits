import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

SPINNER = ['/', '-', '\\', '|']


def eval_adv(model, algo, n_runs=None):
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
            best, rewards = model.play(t)
            algo.observe(rewards[choice])
            regrets.append(rewards[best] - rewards[choice]) # strong regret
            regrets_wk.append(np.max(rewards) - rewards[choice]) # weak regret

        all_regrets.append(np.cumsum(regrets))
        all_regrets_wk.append(np.cumsum(regrets_wk))
    
    all_regrets = np.array(all_regrets)   # (n_runs, T)
    all_regrets_wk = np.array(all_regrets_wk)   # (n_runs, T)
    mean_wk = all_regrets_wk.mean(axis=0)   # (T,)
    std_wk = all_regrets_wk.std(axis=0)     # (T,)
    mean = all_regrets.mean(axis=0)       # (T,)
    std = all_regrets.std(axis=0)         # (T,)
    
    t = np.arange(1, model.T+1)
    
    plt.title(f"Time (X) vs CummReg (Y), {n_runs} iid runs, {model.T} T, {model.K} K")
    plt.plot(t, mean, label='EXP3 Strong Mean Regret', zorder=4)
    plt.plot(t, mean_wk, label='EXP3 Weak Mean Regret', zorder=4)
    plt.fill_between(t, mean - std, mean + std, alpha=0.5, label='±1 std', zorder=3)
    plt.fill_between(t, mean_wk - std_wk, mean_wk + std_wk, alpha=0.5, label='±1 std', zorder=3)
    plt.plot(t, )
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

    