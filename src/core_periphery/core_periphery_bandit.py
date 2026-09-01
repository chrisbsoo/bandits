"""
Core-Periphery Bandit: simulation engine.

Model (5 parameters): mu1_star, mu2_star, v, p, sigma.
  Core (arm 1):      immediate mean = mu1_star always. Available each
                      round i.i.d. with probability (1-p).
  Periphery (arm 2):  always available. immediate mean
                      mu2(n2) = mu2_star * (1 - exp(-v*n2)), n2 = own pull count.
  Reward:             true mean + N(0, sigma), shared sigma across arms.

Regret: pseudo-regret against the TRUE optimal policy pi*, computed via
exact backward induction (Section "The optimal policy"), coupled to the
SAME per-round core-availability draws as the evaluated policy. NOT the
self-referential definition (see "Why Conventional Regret Fails").

Only the Monte Carlo loop is parallelized (prange). The oracle DP table
is computed once per parameter set, sequentially, then reused across all
MC replications and passed in as a fixed input array.
"""

import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads, config


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
    V = np.zeros(T + 2, dtype=np.float64)        # V[n2] for t+1, reused as we sweep t backward
    V_next = np.zeros(T + 2, dtype=np.float64)
    action_table = np.ones((T + 1, T + 1), dtype=np.int8)  # default 1, overwritten where reachable

    # V_next currently represents V_{T+1}(.) = 0 (already zero-initialized)
    for t in range(T, 0, -1):
        V_cur = np.zeros(T + 2, dtype=np.float64)
        max_n = t  # n2 can't exceed number of rounds elapsed so far (t-1 at most before this round);
                   # using t as a safe inclusive upper bound
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
            unavail_val = m2 + V_next[min(n2 + 1, T + 1)]  # core unavailable -> forced periphery
            V_cur[n2] = p * unavail_val + (1.0 - p) * best_avail
        V_next = V_cur

    return action_table


# ----------------------------------------------------------------------
# Single replication: runs the evaluated policy AND the oracle through
# the SAME realized core-availability sequence, returns coupled
# pseudo-regret trajectory.
# ----------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def simulate_run(policy_id, T, mu1_star, mu2_star, v, p, sigma, seed,
                  oracle_action_table, ucb_c=1.0, explore_m=0,
                  window=100, min_floor=20):
    np.random.seed(seed)

    # ---- evaluated policy state ----
    n1 = 0
    n2 = 0
    sum1 = 0.0
    sum2 = 0.0

    # ---- ring buffer for RecencyUCB (policy_id 4): tracks only the most
    # recent `window` periphery rewards, since the periphery's true mean is
    # rising -- a full-history average is systematically dragged down by
    # its own early, low-value pulls.
    peri_buf = np.zeros(window, dtype=np.float64)
    peri_buf_sum = 0.0

    # ---- oracle state (independent trajectory, same availability draws) ----
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
            m1 = (sum1 / n1) if n1 > 0 else BIG
            m2 = (sum2 / n2) if n2 > 0 else BIG
            if core_avail:
                choice = 1 if m1 >= m2 else 2
            else:
                choice = 2

        elif policy_id == 1:  # Explore-then-Commit, force m pulls each, then greedy on all data
            if core_avail and n1 < explore_m:
                choice = 1
            elif n2 < explore_m:
                choice = 2
            elif core_avail:
                m1 = sum1 / n1
                m2 = sum2 / n2
                choice = 1 if m1 >= m2 else 2
            else:
                choice = 2

        elif policy_id == 2:  # UCB1, adapted to availability (local clock: rounds each arm was AVAILABLE)
            if core_avail and n1 == 0:
                choice = 1
            elif n2 == 0:
                choice = 2
            else:
                m2 = sum2 / n2
                ucb2 = m2 + ucb_c * np.sqrt(2.0 * np.log(t + 2.0) / n2)
                if core_avail:
                    m1 = sum1 / n1
                    ucb1 = m1 + ucb_c * np.sqrt(2.0 * np.log(t + 2.0) / n1)
                    choice = 1 if ucb1 >= ucb2 else 2
                else:
                    choice = 2

        elif policy_id == 4:  # RecencyUCB (proposed): min-sample floor (robustness,
            # per the tail-risk finding) + recency-weighted mean for the periphery
            # (tracks its rising true mean instead of being dragged down by early
            # low pulls) + standard UCB exploration bonus on top of both.
            if core_avail and n1 < min_floor:
                choice = 1
            elif n2 < min_floor:
                choice = 2
            elif core_avail:
                m1 = sum1 / n1
                recent_n = n2 if n2 < window else window
                m2_recent = peri_buf_sum / recent_n   # windowed MEAN: tracks rising true value
                ucb1 = m1 + ucb_c * np.sqrt(2.0 * np.log(t + 2.0) / n1)
                # bonus uses the TRUE n2, not the windowed count -- so
                # exploration genuinely tapers off as real evidence
                # accumulates, instead of being perpetually capped at
                # "only 100 samples' worth" of confidence forever
                ucb2 = m2_recent + ucb_c * np.sqrt(2.0 * np.log(t + 2.0) / n2)
                choice = 1 if ucb1 >= ucb2 else 2
            else:
                choice = 2

        else:  # policy_id == 3, Thompson Sampling, Gaussian known-variance conjugate
            if core_avail:
                if n1 > 0:
                    m1 = sum1 / n1
                    s1 = np.random.normal(m1, sigma / np.sqrt(n1))
                else:
                    s1 = np.random.normal(0.0, BIG)
            if n2 > 0:
                m2 = sum2 / n2
                s2 = np.random.normal(m2, sigma / np.sqrt(n2))
            else:
                s2 = np.random.normal(0.0, BIG)
            if core_avail:
                choice = 1 if s1 >= s2 else 2
            else:
                choice = 2
        # ---------------- pull & reward ----------------
        if choice == 1:
            r = np.random.normal(mu1_star, sigma)
            sum1 += r
            n1 += 1
            true_mean = mu1_star
        else:
            r = np.random.normal(m2_now, sigma)
            sum2 += r
            # ring buffer update: overwrite the slot that's `window` pulls old,
            # subtract what it held, add the new reward -- O(1) per step
            slot = n2 % window
            peri_buf_sum += r - peri_buf[slot]
            peri_buf[slot] = r
            n2 += 1
            true_mean = m2_now

        running_regret += (mu_star_chosen - true_mean)
        cum_regret[t] = running_regret
        chosen[t] = choice

    return cum_regret, chosen


# ----------------------------------------------------------------------
# Parallel Monte Carlo driver. Oracle table is computed ONCE outside
# this function and passed in, reused across all replications.
# ----------------------------------------------------------------------
@njit(cache=True, parallel=True)
def monte_carlo(policy_id, n_mc, T, mu1_star, mu2_star, v, p, sigma,
                 base_seed, oracle_action_table, ucb_c=1.0, explore_m=0,
                 n_sample=150, window=100, min_floor=20):
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
                               window, min_floor)
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

            if rule_id == 0:  # oracle: maximize
                best_avail = val_core if val_core >= val_peri else val_peri
            elif rule_id == 1:  # myopic: compare immediate means only
                best_avail = val_core if mu1_star >= m2 else val_peri
            elif rule_id == 2:  # always play core when available
                best_avail = val_core
            else:  # always play periphery even if core available
                best_avail = val_peri

            unavail_val = m2 + V_next[min(n2 + 1, T + 1)]  # forced periphery if core unavailable
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


# Policy IDs, for reference when calling monte_carlo:
POLICY_GREEDY = 0
POLICY_ETC = 1        # pass explore_m explicitly
POLICY_UCB1 = 2
POLICY_THOMPSON = 3
POLICY_RECENCY_UCB = 4  # proposed: pass window, min_floor explicitly