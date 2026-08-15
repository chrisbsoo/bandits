
import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads


@njit(cache=True, fastmath=True)
def simulate_run(policy_id, T, mu1, sigma1, p_sleep, alpha,
                  mu2_star, v, sigma2, ucb_c, seed):
    np.random.seed(seed)

    sleep_remaining = 0   # 0 = arm1 awake; >0 = rounds left asleep
    n1 = 0
    n2 = 0
    sum1 = 0.0
    sum2 = 0.0

    cum_regret = np.empty(T, dtype=np.float64)
    chosen = np.empty(T, dtype=np.int8)
    running_regret = 0.0

    BIG = 1.0e6  # "optimistic infinity" / uninformative-prior scale

    for t in range(T):
        if sleep_remaining == 0:
            if np.random.random() < p_sleep:
                sleep_remaining = alpha  # asleep starting THIS round, for alpha rounds total
        arm1_avail = (sleep_remaining == 0)

        mu2_now = mu2_star * (1.0 - np.exp(-v * n2))

        if arm1_avail:
            best_avail = mu1 if mu1 > mu2_now else mu2_now
        else:
            best_avail = mu2_now

        if policy_id == 0:  # Greedy w/ optimistic init
            m1 = (sum1 / n1) if n1 > 0 else BIG
            m2 = (sum2 / n2) if n2 > 0 else BIG
            if arm1_avail:
                choice = 1 if m1 >= m2 else 2
            else:
                choice = 2

        elif policy_id == 1:  # UCB1 (adapted to availability)
            if arm1_avail and n1 == 0:
                choice = 1
            elif n2 == 0:
                choice = 2
            else:
                m2 = sum2 / n2
                ucb2 = m2 + ucb_c * np.sqrt(2.0 * np.log(t + 2.0) / n2)
                if arm1_avail:
                    m1 = sum1 / n1
                    ucb1 = m1 + ucb_c * np.sqrt(2.0 * np.log(t + 2.0) / n1)
                    choice = 1 if ucb1 >= ucb2 else 2
                else:
                    choice = 2

        else: 
            if arm1_avail:
                if n1 > 0:
                    m1 = sum1 / n1
                    s1 = np.random.normal(m1, sigma1 / np.sqrt(n1))
                else:
                    s1 = np.random.normal(0.0, BIG)
            if n2 > 0:
                m2 = sum2 / n2
                s2 = np.random.normal(m2, sigma2 / np.sqrt(n2))
            else:
                s2 = np.random.normal(0.0, BIG)

            if arm1_avail:
                choice = 1 if s1 >= s2 else 2
            else:
                choice = 2

        if choice == 1:
            r = np.random.normal(mu1, sigma1)
            sum1 += r
            n1 += 1
            true_mean = mu1
        else:
            r = np.random.normal(mu2_now, sigma2)
            sum2 += r
            n2 += 1
            true_mean = mu2_now

        running_regret += (best_avail - true_mean)
        cum_regret[t] = running_regret
        chosen[t] = choice

        if sleep_remaining > 0:
            sleep_remaining -= 1

    return cum_regret, chosen


@njit(cache=True, parallel=True)
def monte_carlo_agg(policy_id, n_mc, T, mu1, sigma1, p_sleep, alpha,
                     mu2_star, v, sigma2, ucb_c, base_seed, n_sample=150):
    sum_regret = np.zeros(T, dtype=np.float64)
    sumsq_regret = np.zeros(T, dtype=np.float64)
    sum_arm2 = np.zeros(T, dtype=np.float64)
    final_regret = np.zeros(n_mc, dtype=np.float64)
    n_sample = min(n_sample, n_mc)
    sample_regret = np.zeros((n_sample, T), dtype=np.float64)

    for i in prange(n_mc):
        seed_i = base_seed + i * 7919 + 1
        cr, ch = simulate_run(policy_id, T, mu1, sigma1, p_sleep, alpha,
                               mu2_star, v, sigma2, ucb_c, seed_i)
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
    import numba
    max_available = numba.config.NUMBA_DEFAULT_NUM_THREADS
    n = max(1, min(n_threads, max_available))
    set_num_threads(n)
    return get_num_threads()
