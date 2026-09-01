"""
Local, single-seed sanity check -- mirrors modal_sweep.py's grid and
policy_specs exactly, but runs directly on your machine (no Modal), to
verify everything (esp. the new ETC formula) works before spending
Modal budget on the full multi-seed run.
"""
import time
import numpy as np
from core_periphery_bandit import (
    compute_oracle_table, monte_carlo, configure_threads,
    POLICY_GREEDY, POLICY_ETC, POLICY_UCB1, POLICY_THOMPSON,
)

configure_threads(8)

mu1_star, mu2_star, sigma = 0.75, 0.6, 0.2   # Case 1
T, n_mc = 5000, 2000
seed = 2026

V_GRID = np.geomspace(0.0002, 0.05, 16)
P_GRID = np.linspace(0.1, 0.9, 16)
T_23 = T ** (2/3)

policy_specs = [
    ("Greedy",   POLICY_GREEDY, {}),
    ("ETC(c=1)", POLICY_ETC, dict(explore_m=int(round(1.0*T_23)))),
    ("ETC(c=2)", POLICY_ETC, dict(explore_m=int(round(2.0*T_23)))),
    ("ETC(c=3)", POLICY_ETC, dict(explore_m=int(round(3.0*T_23)))),
    ("UCB1",     POLICY_UCB1, {}),
    ("Thompson", POLICY_THOMPSON, {}),
]

n_v, n_p = len(V_GRID), len(P_GRID)
results_arr = {name: np.zeros((1, n_v, n_p)) for name, _, _ in policy_specs}

t0 = time.time()
for i, v in enumerate(V_GRID):
    for j, p in enumerate(P_GRID):
        table = compute_oracle_table(mu1_star, mu2_star, v, p, T)
        for name, pid, extra in policy_specs:
            _, _, _, final_r, _ = monte_carlo(pid, n_mc, T, mu1_star, mu2_star, v, p, sigma, seed, table, n_sample=2, **extra)
            results_arr[name][0, i, j] = final_r.mean()
    print(f"  v={v:.5f} row done ({i+1}/{n_v}, {time.time()-t0:.1f}s elapsed)")

print(f"finished in {time.time()-t0:.1f}s")
np.savez("./local_case2_1seed_test.npz", v_grid=V_GRID, p_grid=P_GRID, seeds=np.array([seed]),
         **{name: grid for name, grid in results_arr.items()})
print("saved local_case2_1seed_test.npz")