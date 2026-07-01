import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

SPINNER = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
sizes = {
    (1, 0, 0): (6, 4),
    (0, 1, 0): (6, 4),
    (0, 0, 1): (10, 4),
    (1, 1, 0): (12, 4),
    (1, 0, 1): (10, 7),
    (0, 1, 1): (10, 7),
    (1, 1, 1): (12, 8),
}

def plot(model, n_runs=1, regs=None, wk_regs=None, sleeping=None):
    key = (regs is not None, wk_regs is not None, sleeping is not None)
    fig = plt.figure(figsize=sizes[key])
    n_top = (regs is not None) + (wk_regs is not None)
    has_bottom = sleeping is not None
    nrows = 1 + has_bottom  # 1 if no sleeping, 2 if sleeping
    ncols = max(1, n_top)   # 1 if only sleeping, 2 if any top plots
    gs = fig.add_gridspec(nrows, ncols, hspace=0.35, wspace=0.3)
    t = np.arange(1, model.T+1)

    fig.suptitle(f"K = {model.K}, T = {model.T}, Runs = {n_runs}", fontsize=14, y=1.05 - (0.05 * nrows))

    if regs is not None:
        ax1 = fig.add_subplot(gs[0, 0])
        regs = np.array(regs)   # (n_runs, T)
        mean, std = regs.mean(axis=0), regs.std(axis=0)
        plot_reg(ax1, t, mean, std, "Strong")

    if wk_regs is not None:
        col = 1 if regs is not None else 0
        ax2 = fig.add_subplot(gs[0, col])
        wk_regs = np.array(wk_regs)   # (n_runs, T)
        mean_wk, std_wk = wk_regs.mean(axis=0), wk_regs.std(axis=0)     # (T,)
        plot_reg(ax2, t, mean_wk, std_wk, "Weak")
    
    if sleeping is not None:
        row = 1 if regs is not None else 0
        ax3 = fig.add_subplot(gs[row, :])
        hm = np.array(sleeping)
        plot_arm_aval(ax3, fig, hm, model.T, model.K)

    plt.show()

def plot_reg(ax, t, mean, std, title):
    ax.set_title(f"Cumulative {title} Regret")
    ax.plot(t, mean, label='Mean Regret', zorder=4)
    ax.fill_between(t, mean - std, mean + std, alpha=0.5, label='±1 std', zorder=3)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Regret')
    ax.grid(True, alpha=0.3)
    ax.legend()

def plot_arm_aval(ax, fig, hm, T, K):
    ax.set_title("Arm Availability Heatmap")
    im = ax.imshow(hm, cmap='coolwarm', origin='lower', aspect='auto')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.linspace(0, T-1, 6, dtype=int))
    ax.set_yticks(range(K))
    ax.set_xlabel('Time')
    ax.set_ylabel('Arm')

    