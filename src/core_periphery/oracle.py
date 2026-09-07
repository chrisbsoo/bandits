"""
Oracle: exact backward induction for the optimal policy pi*, plus
sanity-check utilities used to verify the oracle is correct.

Self-contained module -- no shared mutable state with the policy engine
in core_periphery_bandit.py, so this was the safe first piece to extract
into its own file.
"""

import numpy as np
from numba import njit, prange


# ----------------------------------------------------------------------
# Oracle: exact backward induction, O(T^2) time, computed ONCE per
# (mu1_star, mu2_star, v, p, T). Not parallelized -- inherently sequential
# over t (each t needs t+1's results).
# ----------------------------------------------------------------------
@njit(cache=True)
def compute_oracle_table(mu1_star, mu2_star, v, p, T):
    """Returns action_table[t, n2] in {1,2}: optimal action at round t
    (1-indexed) given periphery pull count n2, ASSUMING core is available
    that round (irrelevant/unused entries when core unavailable, since
    action is then forced to 2 regardless)."""
    V_next = np.zeros(T + 2, dtype=np.float64)
    action_table = np.ones((T + 1, T + 1), dtype=np.int8)  # default 1, overwritten where reachable

    for t in range(T, 0, -1):
        V_cur = np.zeros(T + 2, dtype=np.float64)
        max_n = t
        for n2 in range(0, max_n + 1):
            m2 = mu2_star * (1.0 - np.exp(-v * n2))
            val_core = mu1_star + V_next[n2]
            val_peri = m2 + V_next[min(n2 + 1, T + 1)]
            if val_core >= val_peri:
                action_table[t, n2] = 1
                best_avail = val_core
            else:
                action_table[t, n2] = 2
                best_avail = val_peri
            unavail_val = m2 + V_next[min(n2 + 1, T + 1)]
            V_cur[n2] = p * unavail_val + (1.0 - p) * best_avail
        V_next = V_cur

    return action_table


# ----------------------------------------------------------------------
# Sanity-check utility: exact value (via the SAME backward-induction
# recursion) of a few named fixed decision rules, for comparison against
# the oracle's value. Not used in the main experiment pipeline -- only
# for verifying the oracle is actually optimal.
# ----------------------------------------------------------------------
@njit(cache=True)
def compute_rule_value(mu1_star, mu2_star, v, p, T, rule_id):
    """rule_id: 0=oracle (maximizing), 1=myopic (compare immediate means
    only, ignore continuation value), 2=always_core, 3=always_periphery.
    Returns V_1(0): total expected reward from round 1, periphery count 0."""
    V_next = np.zeros(T + 2, dtype=np.float64)

    for t in range(T, 0, -1):
        V_cur = np.zeros(T + 2, dtype=np.float64)
        for n2 in range(0, t + 1):
            m2 = mu2_star * (1.0 - np.exp(-v * n2))
            val_core = mu1_star + V_next[n2]
            val_peri = m2 + V_next[min(n2 + 1, T + 1)]

            if rule_id == 0:
                best_avail = val_core if val_core >= val_peri else val_peri
            elif rule_id == 1:
                best_avail = val_core if mu1_star >= m2 else val_peri
            elif rule_id == 2:
                best_avail = val_core
            else:
                best_avail = val_peri

            unavail_val = m2 + V_next[min(n2 + 1, T + 1)]
            V_cur[n2] = p * unavail_val + (1.0 - p) * best_avail
        V_next = V_cur

    return V_next[0]


# ----------------------------------------------------------------------
# Sanity-check utility: forward-simulate the oracle table using REALIZED
# noisy rewards (not true means), to check empirical average total
# reward converges to compute_oracle_table's implied V_1(0).
# ----------------------------------------------------------------------
@njit(cache=True, parallel=True)
def simulate_oracle_realized_reward(n_mc, T, mu1_star, mu2_star, v, p, sigma,
                                     base_seed, oracle_action_table):
    totals = np.zeros(n_mc, dtype=np.float64)
    for i in prange(n_mc):
        np.random.seed(base_seed + i * 7919 + 1)
        n2_star = 0
        total = 0.0
        for t in range(T):
            core_avail = np.random.random() < (1.0 - p)
            if core_avail:
                a_star = oracle_action_table[min(t + 1, oracle_action_table.shape[0] - 1),
                                              min(n2_star, oracle_action_table.shape[1] - 1)]
            else:
                a_star = 2
            if a_star == 1:
                total += np.random.normal(mu1_star, sigma)
            else:
                m2 = mu2_star * (1.0 - np.exp(-v * n2_star))
                total += np.random.normal(m2, sigma)
                n2_star += 1
        totals[i] = total
    return totals