import time
import numpy as np
import modal

app = modal.App("core-periphery-bandit-sweep")

image = (
    modal.Image.debian_slim()
    .pip_install("numpy", "numba")
    .add_local_python_source("core_periphery_bandit", "oracle_cache", "oracle")
)

oracle_volume = modal.Volume.from_name("core-periphery-oracle-cache", create_if_missing=True)
VOLUME_MOUNT_PATH = "/cache"


@app.function(image=image, cpu=8, timeout=600, volumes={VOLUME_MOUNT_PATH: oracle_volume})
def sweep_cell(mu1_star, mu2_star, v, p, T, n_mc, sigma, policy_specs, base_seed):
    from core_periphery_bandit import monte_carlo, configure_threads
    from oracle_cache import get_oracle_table

    configure_threads(8)  # matches the cpu= requested above
    table = get_oracle_table(mu1_star, mu2_star, v, p, T, cache_dir=VOLUME_MOUNT_PATH)
    oracle_volume.commit()  # flush this container's write so OTHER containers/future runs can see it

    results = {}
    for name, pid, extra in policy_specs:
        mean_r, std_r, frac2, final_r, sample_r = monte_carlo(
            pid, n_mc, T, mu1_star, mu2_star, v, p, sigma, base_seed, table,
            n_sample=2, **extra
        )
        results[name] = float(final_r.mean())
    return (v, p, base_seed, results)


@app.local_entrypoint()
def main(n_seeds: int = 100, seed_start: int = 0, output: str = "modal_multiseed_results.npz"):
    from core_periphery_bandit import POLICY_GREEDY, POLICY_ETC, POLICY_UCB1, POLICY_THOMPSON

    # ---- scenario (edit as needed) ----
    mu1_star, mu2_star, sigma = 0.6, 0.75, 0.2
    T, n_mc = 5000, 2000

    V_GRID = np.geomspace(0.0002, 0.05, 16)
    P_GRID = np.linspace(0.1, 0.9, 16)
    SEEDS = [seed_start + 1000 * i for i in range(n_seeds)]  # widely spaced, avoids seed-stream overlap

    T_23 = T ** (2/3)  
    policy_specs = [
        ("Greedy",       POLICY_GREEDY, {}),
        ("ETC(c=1)",     POLICY_ETC, dict(explore_m=int(round(1.0 * T_23)))),
        ("ETC(c=2)",     POLICY_ETC, dict(explore_m=int(round(2.0 * T_23)))),
        ("ETC(c=3)",     POLICY_ETC, dict(explore_m=int(round(3.0 * T_23)))),
        ("UCB1",         POLICY_UCB1, {}),
        ("Thompson",     POLICY_THOMPSON, {})
    ]
    
    cells = [(mu1_star, mu2_star, v, p, T, n_mc, sigma, policy_specs, seed)
             for seed in SEEDS for v in V_GRID for p in P_GRID]

    print(f"Dispatching {len(cells)} tasks ({n_seeds} seeds x {len(V_GRID)}x{len(P_GRID)} grid) to Modal...")
    t0 = time.time()

    n_v, n_p = len(V_GRID), len(P_GRID)
    # results[policy_name] has shape (n_seeds, n_v, n_p)
    results_arr = {name: np.zeros((n_seeds, n_v, n_p)) for name, _, _ in policy_specs}

    v_index = {v: i for i, v in enumerate(V_GRID)}
    p_index = {p: j for j, p in enumerate(P_GRID)}
    seed_index = {seed: k for k, seed in enumerate(SEEDS)}

    for v, p, seed, results in sweep_cell.starmap(cells):
        i, j, k = v_index[v], p_index[p], seed_index[seed]
        for name, val in results.items():
            results_arr[name][k, i, j] = val

    print(f"Finished in {time.time()-t0:.1f}s (wall clock, across all parallel containers)")

    np.savez(output, v_grid=V_GRID, p_grid=P_GRID, seeds=np.array(SEEDS),
             **{name: grid for name, grid in results_arr.items()})
    print(f"saved {output}  (shape per policy: {n_seeds} seeds x {n_v} x {n_p})")