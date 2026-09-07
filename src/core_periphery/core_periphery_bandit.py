import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads, config
from oracle import compute_oracle_table, compute_rule_value, simulate_oracle_realized_reward  # noqa: F401  (re-exported)

from algorithms.greedy import naive_greedy, greedy_etc
from algorithms.bandits import bandits_ucb, bandits_thompson


# ----------------------------------------------------------------------
# Single replication: runs the evaluated policy AND the oracle through
# the SAME realized core-availability sequence, returns coupled
# pseudo-regret trajectory.
# ----------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def simulate_run(policy_id, T, mu1_star, mu2_star, v, p, sigma, seed,
                  oracle_action_table, ucb_c=1.0, explore_m=0,
                  window=100, min_floor=20, eps=0.5):
    np.random.seed(seed)

    # ---- evaluated policy state ----
    n1 = 0
    n2 = 0
    sum1 = 0.0
    sum2 = 0.0
    peri_buf = np.zeros(window, dtype=np.float64)
    peri_buf_sum = 0.0
    peri_history = np.zeros(T, dtype=np.float64)
    growing_window_sum = 0.0
    left_idx = 0
    committed = 0
    n2_star = 0

    cum_regret = np.empty(T, dtype=np.float64)
    chosen = np.empty(T, dtype=np.int8)
    running_regret = 0.0

    BIG = 1.0e6  # optimistic init for Greedy/ETC pre-explore phase

    for t in range(T):
        core_avail = np.random.random() < (1.0 - p)   # SAME draw feeds both policy and oracle below

        # ---------------- oracle's own decision ----------------
        if core_avail:
            a_star = oracle_action_table[min(t + 1, oracle_action_table.shape[0] - 1),
                                          min(n2_star, oracle_action_table.shape[1] - 1)]
        else:
            a_star = 2
        if a_star == 1:
            mu_star_chosen = mu1_star
        else:
            mu_star_chosen = mu2_star * (1.0 - np.exp(-v * n2_star))
            n2_star += 1

        # ---------------- evaluated policy's decision ----------------
        m2_now = mu2_star * (1.0 - np.exp(-v * n2))

        if policy_id == 0:  # Greedy, optimistic init
            choice = naive_greedy(sum1, n1, sum2, n2, BIG, core_avail)

        elif policy_id == 1:  # Explore-then-Commit, force m pulls each, then greedy on all data
            choice = greedy_etc(sum1, n1, sum2, n2, core_avail, explore_m)

        elif policy_id == 2:  # UCB1, adapted to availability (local clock: rounds each arm was AVAILABLE)
            choice = bandits_ucb(sum1, n1, sum2, n2, t, core_avail, ucb_c)

        elif policy_id == 3:
            choice = bandits_thompson(sum1, n1, sum2, n2, sigma, BIG, core_avail)

        
        # ---------------- pull & reward ----------------
        if choice == 1:
            r = np.random.normal(mu1_star, sigma)
            sum1 += r
            n1 += 1
            true_mean = mu1_star
        else:
            r = np.random.normal(m2_now, sigma)
            sum2 += r
            slot = n2 % window
            peri_buf_sum += r - peri_buf[slot]
            peri_buf[slot] = r
            peri_history[n2] = r
            growing_window_sum += r
            n2 += 1
            w_new = int(n2 * eps)
            if w_new < 1:
                w_new = 1
            new_start = n2 - w_new
            while left_idx < new_start:
                growing_window_sum -= peri_history[left_idx]
                left_idx += 1
            true_mean = m2_now

        running_regret += (mu_star_chosen - true_mean)
        cum_regret[t] = running_regret
        chosen[t] = choice

    return cum_regret, chosen


@njit(cache=True, parallel=True)
def monte_carlo(policy_id, n_mc, T, mu1_star, mu2_star, v, p, sigma,
                 base_seed, oracle_action_table, ucb_c=1.0, explore_m=0,
                 n_sample=150, window=100, min_floor=20, eps=0.5):
    n_sample = min(n_sample, n_mc)
    sum_regret = np.zeros(T, dtype=np.float64)
    sumsq_regret = np.zeros(T, dtype=np.float64)
    sum_arm2 = np.zeros(T, dtype=np.float64)
    final_regret = np.zeros(n_mc, dtype=np.float64)
    sample_regret = np.zeros((n_sample, T), dtype=np.float64)

    for i in prange(n_mc):
        seed_i = base_seed + i * 7919 + 1
        cr, ch = simulate_run(policy_id, T, mu1_star, mu2_star, v, p, sigma,
                               seed_i, oracle_action_table, ucb_c, explore_m,
                               window, min_floor, eps)
        sum_regret += cr
        sumsq_regret += cr * cr
        for t in range(T):
            if ch[t] == 2:
                sum_arm2[t] += 1.0
        final_regret[i] = cr[T - 1]
        if i < n_sample:
            sample_regret[i, :] = cr

    mean_regret = sum_regret / n_mc
    var_regret = sumsq_regret / n_mc - mean_regret ** 2
    std_regret = np.sqrt(np.maximum(var_regret, 0.0))
    frac_arm2 = sum_arm2 / n_mc

    return mean_regret, std_regret, frac_arm2, final_regret, sample_regret


def configure_threads(n_threads=8):
    max_available = config.NUMBA_DEFAULT_NUM_THREADS
    n = max(1, min(n_threads, max_available))
    set_num_threads(n)
    return get_num_threads()


# Policy IDs, for reference when calling monte_carlo:
POLICY_GREEDY = 0
POLICY_ETC = 1        
POLICY_UCB1 = 2
POLICY_THOMPSON = 3