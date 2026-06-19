import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

SPINNER = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

def plot(n_runs, model, all_regrets, all_regrets_wk, show=2, hm=None):
    all_regrets = np.array(all_regrets)   # (n_runs, T)
    all_regrets_wk = np.array(all_regrets_wk)   # (n_runs, T)
    mean_wk = all_regrets_wk.mean(axis=0)   # (T,)
    std_wk = all_regrets_wk.std(axis=0)     # (T,)
    mean = all_regrets.mean(axis=0)       # (T,)
    std = all_regrets.std(axis=0)         # (T,)
    
    t = np.arange(1, model.T+1)

    nump = 1
    if hm is not None:
        nump += 1

    fig, ax = plt.subplots(1, nump, figsize=(12, 5))

    i = 0
    
    ax[i].set_title(f"Time (X) vs CummReg (Y), {n_runs} iid runs, {model.T} T, {model.K} K")
    if show == 1 or show == 2:
        ax[i].plot(t, mean, label='Strong Mean Regret', zorder=4)
        ax[i].fill_between(t, mean - std, mean + std, alpha=0.5, label='±1 std', zorder=3)
    if show == 0 or show == 2:
        ax[i].plot(t, mean_wk, label='Weak Mean Regret', zorder=4)
        ax[i].fill_between(t, mean_wk - std_wk, mean_wk + std_wk, alpha=0.5, label='±1 std', zorder=3)
    ax[i].set_xlabel('Time')
    ax[i].set_ylabel('Cumulative Regret')
    ax[i].legend()

    i += 1
    if i < nump:
        ax[i].set_title(f"Arm Availability Heatmap, {model.T} T, {model.K} K")
        im = ax[i].imshow(hm, cmap="coolwarm", origin="lower", aspect='auto')
        fig.colorbar(im, ax=ax[i])
        ax[i].set_xticks(np.linspace(0, model.T-1, 6, dtype=int))
        ax[i].set_yticks(range(model.K))
        ax[i].set_xlabel('Time')
        ax[i].set_ylabel('Arm Availability')

    plt.show()
    

    