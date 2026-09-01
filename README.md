# Core-Periphery Bandits

A two-armed bandit model of vacancy allocation in online labour markets: an exogenously unavailable, stable arm (**core**, experienced workers) against an always-available arm whose value rises with use (**periphery**, novice workers). Built to study when exploration algorithms correctly discover that a currently-worse option is actually the better long-run choice, and to design one that does this reliably.

See results + notebook here: https://github.com/chrisbsoo/core-periphery/blob/main/src/core_periphery/sig_analysis.ipynb

## Motivation

Online labour platforms must repeatedly decide whether a new vacancy goes to a small, experienced worker pool or a large, novice one. The core is stable but frequently unavailable; the periphery is always available, but its true productivity is only revealed, and only improves, as workers are actually given the chance to work. Pallais (2014, *AER*) documents exactly this failure empirically in a field experiment on oDesk/Upwork: novice workers were systematically under-hired relative to their true, initially-unrevealed ability. This project formalizes the algorithmic version of that problem.

## Model

Five parameters: `μ1*, μ2*, v, p, σ`.

- **Core**: fixed mean `μ1*`, available each round independently with probability `1-p`.
- **Periphery**: always available, mean `μ2(n) = μ2*(1 - e^(-vn))`, `n` = its own pull count, a standard saturating rising-bandit curve. `μ2*` and `v` are unknown to any algorithm.
- **Noise**: shared `σ` across both arms.

## Key findings

**A "natural" regret definition is provably gameable.** Comparing a policy against a benchmark built from its own pull-count trajectory lets a policy that never touches the periphery score exactly zero regret, regardless of how good the periphery secretly is. Fixed by comparing against an independently-computed optimal policy (`π*`) instead, coupled to the same exogenous core-availability draws, but with its own separate pull-count trajectory.

**The "obvious" optimal policy is itself wrong.** Exact backward induction shows the true optimal policy sometimes deliberately plays the currently-worse arm, sacrificing near-term reward because it raises the periphery's pull count, paying off over the remaining horizon. Formalized as **Proposition 1**: the optimal policy deviates from myopic behavior if and only if `μ2* > μ1*` and enough horizon remains to recoup the investment.

**The oracle is verified correct, not assumed**: exact agreement (machine precision) against an independent brute-force implementation, both at single points and across every one of 1,890 reachable states in a full-table check; dominates 51 randomized policies and both fixed extremes in a sweep; forward-simulated realized rewards converge to the DP-predicted value under Monte Carlo.

**No single baseline dominates.** A 256-cell `(v, p)` sweep, 100 seeds, paired-significance-tested, produces a "winner map", different baselines own distinct, mechanistically-explainable regions, driven by two separate mechanisms: **investment timing** (does the periphery reveal itself fast enough, relative to `p` and remaining horizon, to be worth chasing) and **estimation robustness** (a completely separate failure mode where naive Greedy exhibits a rare but catastrophic regret tail from committing off a single noisy sample of a rarely-available arm, up to 300x its typical outcome in ~0.1% of runs).

**Explore-then-Commit's exploration budget is theory-grounded**: sized as `m = c · T^(2/3)`, the minimax-optimal exploration length for a stationary two-armed bandit with unknown gap (Lattimore & Szepesvári), with `c` calibrated empirically since the theorem doesn't account for this environment's rising mean or asymmetric misclassification cost.

## Proposed algorithm: RecencyUCB

Motivated directly by the two mechanisms above. Structurally still a UCB-style optimistic-index policy, but with two different memory strategies per arm: **full-history** mean for the core (stationary, no reason to discard old data) and a **sliding-window** mean for the periphery (non-stationary/rising, old low-value pulls actively mislead a full-history average). A minimum-sample floor (`min_floor=20`) before ever committing directly targets the estimation-robustness failure mode; the windowed mean directly targets the investment-timing failure mode. Exploration bonus uses the *true* pull count (not the windowed count) for both arms, so exploration genuinely tapers off as real evidence accumulates.

## Results

100-seed, 256-cell `(v,p)` sweep, both fixed instances (`μ2* > μ1*` and `μ2* < μ1*`), paired t-test per cell:

| | Case 1 (periphery genuinely better) | Case 2 (periphery genuinely worse) |
|---|---|---|
| **Cells won** | 89/256 (34.8%) — most of any policy | 0/256 |
| **Significant wins** | 89/89 (100%) | — |
| **Grid-wide mean regret** | 99.6 — best of any policy, ~30% below next-best (139.3) | 1.53 — mid-pack; ~10x above best (0.14), but far below worst naive baseline (50.9) |

RecencyUCB wins decisively and confidently in the regime it targets (investment timing under a genuinely-improving arm), at a modest, mechanistically-explained cost in the regime governed by a different mechanism (estimation robustness under a confirmed-inferior, stationary arm), its fixed window leaves a persistent variance floor that full-history estimators don't have once nothing is actually changing.

## Repo structure

core_periphery_bandit.py # simulation engine: oracle DP, all policies (incl. RecencyUCB), parallel Monte Carlo (numba)
oracle_cache.py # disk cache for oracle tables, concurrency-safe (verified under real multiprocessing)
cloud_sweep.py # distributed (v,p,seed) sweep via Modal.com
sig_analysis.py # significance testing, metrics table, winner-map plotting


## Running it

**Locally:**
```python
from core_periphery_bandit import compute_oracle_table, monte_carlo, configure_threads, POLICY_RECENCY_UCB

configure_threads(8)
table = compute_oracle_table(mu1_star=0.6, mu2_star=0.75, v=0.01, p=0.85, T=5000)
mean_r, std_r, frac2, final_r, sample_r = monte_carlo(
    POLICY_RECENCY_UCB, n_mc=2000, T=5000, mu1_star=0.6, mu2_star=0.75, v=0.01, p=0.85,
    sigma=0.2, base_seed=2026, oracle_action_table=table, window=100, min_floor=20,
)
```

**Distributed sweep** (needs a [Modal](https://modal.com) account):

pip install modal
modal setup
modal run --detach modal_sweep.py --n-seeds 100 --output results.npz


**Analysis:**
```python
from sweep_analysis import analyze_sweep
analyze_sweep("results.npz", title="Case 1", save_path="winner_map.png")
```

## Engineering notes

- Simulation core is `numba`-JIT-compiled; only the Monte Carlo loop is parallelized (`prange`), the oracle's backward induction is inherently sequential, `O(T²)`, and cached to disk (and to a persistent Modal Volume) since it depends only on `(μ1*, μ2*, v, p, T)`, never the random seed.
- Cache correctness verified under genuine concurrent access, multiprocessing, and up to 100 parallel Modal containers writing to a shared volume, via exact array-equality checks against ground truth.
- Full sweeps (25,600+ tasks: 100 seeds × 256 cells × 7 policies) run on Modal.com in minutes for a few dollars of CPU-only compute; checkpointed periodically to disk so a network interruption or crash doesn't lose near-complete progress.

## Known limitations

- Significance testing (paired t-test per cell) has no multiple-comparisons correction across the 256 tested cells, and its normality assumption is untested in regions with confirmed skewed regret distributions (the tail-risk finding above).
- `K=2` only; the natural `K=10` extension (matching Upwork's own public "Top Rated = top 10%" designation) is scoped out, exact backward induction becomes intractable with multiple rested/rising arms.
- RecencyUCB's fixed window is a design trade-off, not a free lunch, a growing/adaptive window is a natural next step to close the Case 2 gap without sacrificing Case 1 performance.

## References

- Pallais, A. (2014). Inefficient Hiring in Entry-Level Labor Markets. *American Economic Review*.
- Kleinberg, R., Niculescu-Mizil, A., & Sharma, Y. (2010). Regret Bounds for Sleeping Experts and Bandits.
- Metelli, A. M., Trovò, F., Pirola, M., & Restelli, M. (2022). Stochastic Rising Bandits. *ICML*.
- Basu, S., Sen, R., Sanghavi, S., & Shakkottai, S. (2019). Blocking Bandits. *NeurIPS*.
- Doeringer, P. B., & Piore, M. J. (1971). Internal Labor Markets and Manpower Analysis.
- Lattimore, T., & Szepesvári, C. (2020). Bandit Algorithms. Cambridge University Press.
