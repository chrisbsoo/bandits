"""
UCB1 -- continuous exploration, never fully commits. Adapted to the
two-armed floor/ceiling environment (see environment.py).
"""

import numpy as np
from numba import njit, prange

@njit(cache=True)
def func_target(n, t, L, U, a, b):
    n_term = n ** a if n > 0 else 0.0
    t_term = t ** b if t > 0 else 0.0
    return L + (U - L) / (1.0 + np.exp(-(n_term - t_term)))

@njit(cache=True)
def ucb1_decide(n1, sum1, n2, sum2, t, ucb_c=1.0):
    if n1 == 0:
        return 1
    if n2 == 0:
        return 2
    m1 = sum1 / n1
    m2 = sum2 / n2
    bonus1 = ucb_c * np.sqrt(2.0 * np.log(t + 2.0) / n1)
    bonus2 = ucb_c * np.sqrt(2.0 * np.log(t + 2.0) / n2)
    return 1 if (m1 + bonus1) >= (m2 + bonus2) else 2

@njit(cache=True)
def commit_ucb_decide(n1, sum1, n2, sum2, t, committed, min_floor=5, ucb_c=1.0):
    if committed == 1:
        return 1, 1
    if committed == 2:
        return 2, 2
    if n1 < min_floor:
        return 1, 0
    if n2 < min_floor:
        return 2, 0

    m1 = sum1 / n1
    m2 = sum2 / n2
    bonus1 = ucb_c * np.sqrt(2.0 * np.log(t + 2.0) / n1)
    bonus2 = ucb_c * np.sqrt(2.0 * np.log(t + 2.0) / n2)
    lcb1 = m1 - bonus1
    lcb2 = m2 - bonus2

    if lcb1 > m2:
        return 1, 1  # confidently resolved: commit arm1 forever
    if lcb2 > m1:
        return 2, 2  # confidently resolved: commit arm2 forever

    ucb1_val = m1 + bonus1
    ucb2_val = m2 + bonus2
    choice = 1 if ucb1_val >= ucb2_val else 2
    return choice, 0

@njit(cache=True)
def balanced_explore_decide(n1, sum1, n2, sum2, t, T, c=1.0):
    m = int(c * T ** (2.0 / 3.0))
    if t <= m:
        return 1 if n1 <= n2 else 2
    if n1 == 0:
        return 1
    if n2 == 0:
        return 2
    m1 = sum1 / n1
    m2 = sum2 / n2
    return 1 if m1 >= m2 else 2

@njit(cache=True)
def hybrid_decide(n1, sum1, n2, sum2, t, committed, switches, last_arm, balanced_pulls_done,
                   min_floor=5, balanced_cap=50, switch_threshold=25, ucb_c=1.0):
    if committed == 1:
        return 1, 1, switches, 1, balanced_pulls_done
    if committed == 2:
        return 2, 2, switches, 2, balanced_pulls_done
    if n1 < min_floor:
        return 1, 0, switches, 1, balanced_pulls_done
    if n2 < min_floor:
        return 2, 0, switches, 2, balanced_pulls_done

    m1 = sum1 / n1
    m2 = sum2 / n2
    bonus1 = ucb_c * np.sqrt(2.0 * np.log(t + 2.0) / n1)
    bonus2 = ucb_c * np.sqrt(2.0 * np.log(t + 2.0) / n2)
    lcb1 = m1 - bonus1
    lcb2 = m2 - bonus2

    # resolution check -- fires regardless of phase
    if lcb1 > m2:
        return 1, 1, switches, 1, balanced_pulls_done
    if lcb2 > m1:
        return 2, 2, switches, 2, balanced_pulls_done

    # adaptive balanced phase: keep going (up to the cap) while unresolved
    if balanced_pulls_done < balanced_cap:
        choice = 1 if n1 <= n2 else 2
        new_switches = switches + (1 if (last_arm != 0 and choice != last_arm) else 0)
        return choice, 0, new_switches, choice, balanced_pulls_done + 1

    # main phase: circuit breaker + ordinary UCB comparison
    if switches >= switch_threshold:
        choice = 1 if m1 >= m2 else 2
        return choice, choice, switches, choice, balanced_pulls_done

    ucb1_val = m1 + bonus1
    ucb2_val = m2 + bonus2
    choice = 1 if ucb1_val >= ucb2_val else 2
    new_switches = switches + (1 if (last_arm != 0 and choice != last_arm) else 0)
    return choice, 0, new_switches, choice, balanced_pulls_done

@njit(cache=True)
def rawucb_decide(n1, prefix1, n2, prefix2, t, alpha=1.0, subgaussian=1.0):
    if n1 == 0:
        return 1
    if n2 == 0:
        return 2

    log_term = alpha * np.log(t + 2.0)

    best1 = 1e18
    for w in range(1, n1 + 1):
        windowed_sum = prefix1[n1] - prefix1[n1 - w]
        windowed_mean = windowed_sum / w
        bonus = np.sqrt(2.0 * subgaussian**2 * log_term / w)
        idx = windowed_mean + bonus
        if idx < best1:
            best1 = idx

    best2 = 1e18
    for w in range(1, n2 + 1):
        windowed_sum = prefix2[n2] - prefix2[n2 - w]
        windowed_mean = windowed_sum / w
        bonus = np.sqrt(2.0 * subgaussian**2 * log_term / w)
        idx = windowed_mean + bonus
        if idx < best2:
            best2 = idx

    return 1 if best1 >= best2 else 2


@njit(cache=True)
def eff_rawucb_decide(n1, prefix1, n2, prefix2, t, m_grid=2.0, alpha=1.0, subgaussian=1.0):
    if n1 == 0:
        return 1
    if n2 == 0:
        return 2

    log_term = alpha * np.log(t + 2.0)

    best1 = 1e18
    w = 1.0
    while w <= n1:
        w_int = int(w)
        if w_int < 1:
            w_int = 1
        windowed_sum = prefix1[n1] - prefix1[n1 - w_int]
        windowed_mean = windowed_sum / w_int
        bonus = np.sqrt(2.0 * subgaussian**2 * log_term / w_int)
        idx = windowed_mean + bonus
        if idx < best1:
            best1 = idx
        w *= m_grid
    windowed_mean_full = prefix1[n1] / n1
    bonus_full = np.sqrt(2.0 * subgaussian**2 * log_term / n1)
    idx_full = windowed_mean_full + bonus_full
    if idx_full < best1:
        best1 = idx_full

    best2 = 1e18
    w = 1.0
    while w <= n2:
        w_int = int(w)
        if w_int < 1:
            w_int = 1
        windowed_sum = prefix2[n2] - prefix2[n2 - w_int]
        windowed_mean = windowed_sum / w_int
        bonus = np.sqrt(2.0 * subgaussian**2 * log_term / w_int)
        idx = windowed_mean + bonus
        if idx < best2:
            best2 = idx
        w *= m_grid
    windowed_mean_full2 = prefix2[n2] / n2
    bonus_full2 = np.sqrt(2.0 * subgaussian**2 * log_term / n2)
    idx_full2 = windowed_mean_full2 + bonus_full2
    if idx_full2 < best2:
        best2 = idx_full2

    return 1 if best1 >= best2 else 2

@njit(cache=True, parallel=True)
def run_ucb1_monte_carlo(n_mc, T, L1, U1, a, b, L2, U2, sigma, base_seed, ucb_c=1.0):
    totals = np.zeros(n_mc)
    for i in prange(n_mc):
        np.random.seed(base_seed + i * 7919 + 1)
        n1, n2 = 0, 0
        sum1, sum2 = 0.0, 0.0
        total = 0.0
        for t in range(1, T + 1):
            arm = ucb1_decide(n1, sum1, n2, sum2, t, ucb_c)
            if arm == 1:
                mu = func_target(n1, t, L1, U1, a, b)
                r = np.random.normal(mu, sigma)
                sum1 += r
                n1 += 1
            else:
                mu = func_target(n2, t, L2, U2, a, b)
                r = np.random.normal(mu, sigma)
                sum2 += r
                n2 += 1
            total += r
        totals[i] = total
    return totals

@njit(cache=True, parallel=True)
def run_commit_ucb_monte_carlo(n_mc, T, L1, U1, a, b, L2, U2, sigma, base_seed,
                                 min_floor=5, ucb_c=1.0):
    totals = np.zeros(n_mc)
    for i in prange(n_mc):
        np.random.seed(base_seed + i * 7919 + 1)
        n1, n2 = 0, 0
        sum1, sum2 = 0.0, 0.0
        total = 0.0
        committed = 0
        for t in range(1, T + 1):
            arm, committed = commit_ucb_decide(n1, sum1, n2, sum2, t, committed, min_floor, ucb_c)
            if arm == 1:
                mu = func_target(n1, t, L1, U1, a, b)
                r = np.random.normal(mu, sigma)
                sum1 += r
                n1 += 1
            else:
                mu = func_target(n2, t, L2, U2, a, b)
                r = np.random.normal(mu, sigma)
                sum2 += r
                n2 += 1
            total += r
        totals[i] = total
    return totals

@njit(cache=True, parallel=True)
def run_balanced_explore_monte_carlo(n_mc, T, L1, U1, a, b, L2, U2, sigma, base_seed, c=1.0):
    totals = np.zeros(n_mc)
    for i in prange(n_mc):
        np.random.seed(base_seed + i * 7919 + 1)
        n1, n2 = 0, 0
        sum1, sum2 = 0.0, 0.0
        total = 0.0
        for t in range(1, T + 1):
            arm = balanced_explore_decide(n1, sum1, n2, sum2, t, T, c)
            if arm == 1:
                mu = func_target(n1, t, L1, U1, a, b)
                r = np.random.normal(mu, sigma)
                sum1 += r
                n1 += 1
            else:
                mu = func_target(n2, t, L2, U2, a, b)
                r = np.random.normal(mu, sigma)
                sum2 += r
                n2 += 1
            total += r
        totals[i] = total
    return totals

@njit(cache=True, parallel=True)
def run_hybrid_monte_carlo(n_mc, T, L1, U1, a, b, L2, U2, sigma, base_seed,
                             balanced_cap=50, min_floor=5, switch_threshold=25, ucb_c=1.0):
    totals = np.zeros(n_mc)
    for i in prange(n_mc):
        np.random.seed(base_seed + i * 7919 + 1)
        n1, n2 = 0, 0
        sum1, sum2 = 0.0, 0.0
        committed, switches, last_arm, balanced_pulls_done = 0, 0, 0, 0
        total = 0.0
        for t in range(1, T + 1):
            arm, committed, switches, last_arm, balanced_pulls_done = hybrid_decide(
                n1, sum1, n2, sum2, t, committed, switches, last_arm, balanced_pulls_done,
                min_floor, balanced_cap, switch_threshold, ucb_c
            )
            if arm == 1:
                mu = func_target(n1, t, L1, U1, a, b)
                r = np.random.normal(mu, sigma)
                sum1 += r
                n1 += 1
            else:
                mu = func_target(n2, t, L2, U2, a, b)
                r = np.random.normal(mu, sigma)
                sum2 += r
                n2 += 1
            total += r
        totals[i] = total
    return totals

@njit(cache=True, parallel=True)
def run_rawucb_monte_carlo(n_mc, T, L1, U1, a, b, L2, U2, sigma, base_seed,
                             alpha=1.0, subgaussian=1.0):
    totals = np.zeros(n_mc)
    for i in prange(n_mc):
        np.random.seed(base_seed + i * 7919 + 1)
        n1, n2 = 0, 0
        prefix1 = np.zeros(T + 1)
        prefix2 = np.zeros(T + 1)
        total = 0.0
        for t in range(1, T + 1):
            arm = rawucb_decide(n1, prefix1, n2, prefix2, t, alpha, subgaussian)
            if arm == 1:
                mu = func_target(n1, t, L1, U1, a, b)
                r = np.random.normal(mu, sigma)
                n1 += 1
                prefix1[n1] = prefix1[n1 - 1] + r
            else:
                mu = func_target(n2, t, L2, U2, a, b)
                r = np.random.normal(mu, sigma)
                n2 += 1
                prefix2[n2] = prefix2[n2 - 1] + r
            total += r
        totals[i] = total
    return totals


@njit(cache=True, parallel=True)
def run_eff_rawucb_monte_carlo(n_mc, T, L1, U1, a, b, L2, U2, sigma, base_seed,
                                 m_grid=2.0, alpha=1.0, subgaussian=1.0):
    totals = np.zeros(n_mc)
    for i in prange(n_mc):
        np.random.seed(base_seed + i * 7919 + 1)
        n1, n2 = 0, 0
        prefix1 = np.zeros(T + 1)
        prefix2 = np.zeros(T + 1)
        total = 0.0
        for t in range(1, T + 1):
            arm = eff_rawucb_decide(n1, prefix1, n2, prefix2, t, m_grid, alpha, subgaussian)
            if arm == 1:
                mu = func_target(n1, t, L1, U1, a, b)
                r = np.random.normal(mu, sigma)
                n1 += 1
                prefix1[n1] = prefix1[n1 - 1] + r
            else:
                mu = func_target(n2, t, L2, U2, a, b)
                r = np.random.normal(mu, sigma)
                n2 += 1
                prefix2[n2] = prefix2[n2 - 1] + r
            total += r
        totals[i] = total
    return totals
