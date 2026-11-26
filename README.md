# WSN Sensor Placement: Coverage vs Deployment Cost (PSO vs SQP)

SYDE 411 project code (Group 9). Places N sensors in a 10 × 10 region to
**maximize area coverage while minimizing location-dependent deployment cost**,
subject to **multi-hop connectivity to a fixed base station**, **region
boundaries**, and a **deployment budget**. The problem is solved two ways and
compared:

- **PSO**: hand-written particle swarm optimization. It evaluates the original
  (discontinuous) coverage and cost directly and handles connectivity and budget
  with penalties. Positions are clipped to the region. The reported solution is
  the best feasible layout found.
- **SQP**: SciPy `SLSQP` on a smooth reformulation (sigmoid/noisy-OR coverage,
  sigmoid-weighted Laplacian Fiedler value, smoothed distance cost), wrapped in a
  β-continuation schedule. The reported solution is the best feasible
  continuation stage (see [Solution selection](#solution-selection-and-penalties-revised)).

The technical report (Sections 2–4) is the specification. Every final layout
from either method is re-scored with the **original** model: hard coverage,
exact cost, BFS connectivity with r_c, bounds, and budget. Only layouts that
satisfy all three constraints are counted as feasible.

## Installation

Python ≥ 3.10.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python main.py                                   # baseline: N=10, alpha=0.7, seeds 0-7 (~3 min serial)
python main.py --experiments all --workers 8     # baseline + N sweep + alpha sweep + budget sweep (current results/)
python main.py --experiments nsweep alphasweep   # pick experiments
python main.py --seeds 0 1 2                     # override the seeds
python main.py --quick --experiments all         # smoke test with reduced iterations (NOT report parameters)
python main.py --figures-only --out results      # re-plot from saved CSVs without re-running
python -m pytest                                 # basic sanity checks of the model and solvers
python scripts/verify_results.py results         # independent re-check of every saved layout
```

`scripts/verify_results.py` rebuilds every layout from `positions.csv`: the
reported solution of every run, plus the final-stage layout of every SQP run.
It recomputes coverage, cost, J, bounds, budget, unused budget, and
connectivity with r_c = 1.8 using separately written code: SciPy
`connected_components` instead of the package BFS, and no calls into
`wsn.evaluation`. It then reports any disagreement with `runs.csv`, lists runs
where no feasible solution was found, and prints feasibility, J mean/sd/range,
and runtime per configuration over the seeds.

`--workers` runs independent (N, α, method, seed) runs in parallel processes.
Each process is pinned to one BLAS thread, and times are measured per run inside
the worker. With heavy parallelism, absolute times are somewhat inflated by
contention, but PSO and SQP are affected alike.

## Parameters (report Section 3)

All parameters are centralized in [wsn/config.py](wsn/config.py).

| Quantity | Value |
| --- | --- |
| Region M, base station b | [0,10] × [0,10], b = (5, 5) (communication only, no coverage) |
| Radii | r_s = r_c = 1.8 |
| Sample grid P | 40 × 40 = 1600 points including the boundary (spacing 10/39) |
| Cost k(s) | k0 + w_d κ_d ‖s − b‖ + w_t Σ_j a_j exp(−‖s − μ_j‖² / 2σ_j²) with k0 = 1, κ_d = 0.05, w_d = w_t = 1 |
| Terrain regions (μ, a, σ) | ((2.5, 7.0), 0.8, 1.2), ((7.5, 7.0), 0.6, 1.5), ((6.0, 2.5), 0.7, 1.3) |
| k_max | max over M of k(s) = 1.9644 (dense grid + L-BFGS-B polish; fixed for all runs) |
| Normalized cost | K = K_total / (N k_max) |
| Budget | B(N) = 1.8 N |
| Objective | J = α C − (1 − α) K, baseline α = 0.7 |
| PSO | 40 particles, 120 iterations, w = 0.7, c1 = c2 = 1.5, \|v\| ≤ 0.3 · width, γ = ρ = 2 |
| PSO fitness | F = J − γ φ − ρ max(0, (K_total − B)/B) with φ = N − N_connected (number of disconnected sensors) |
| SQP | β = (1, 3, 8, 20, 50), ≤ 200 SLSQP iterations per stage, finite-difference gradients, ftol 1e-7 |
| SQP surrogates | λ₂(S, β) ≥ ε = 0.05 with r_c_eff = r_c − δ = 1.4; η = 0.01 in the smoothed distance cost |
| Experiments | seeds 0–7; N ∈ {5, 10, 15, 20, 30} at α = 0.7; α ∈ {0, 0.25, 0.5, 0.7, 1} at N = 10 |
| Budget sensitivity | B(N) = b N with b ∈ {1.4, 1.5, 1.6, 1.8} at N = 10, α = 0.7 (1.8 remains the baseline) |
| Budget activity | `active` if unused budget ≤ 1e-6·B, `near` if ≤ 2% of B, otherwise `inactive` (or `violated`) |

Choices the report does not fix, and how they are set:

- **SQP initial guess:** a uniform random layout in M from the run's seed.
- **PSO initialization:** uniform positions in M and uniform velocities in [−v_max, v_max].
- **SLSQP `ftol`:** 1e-7.
- **Numerical tolerances:** 1e-8 on the budget and 1e-9 on the bounds when checking feasibility.

## Code layout

| File | Responsibility |
| --- | --- |
| [wsn/config.py](wsn/config.py) | Frozen `Config` dataclass holding every report parameter. Use `with_overrides(...)` for variants. |
| [wsn/problem.py](wsn/problem.py) | `build_problem(cfg)`: sample grid P, base station, bounds, k_max (cached). |
| [wsn/geometry.py](wsn/geometry.py) | Hard coverage C(S); BFS reachability from b; smooth coverage C̄(S, β) (computed in log space with `log_expit`); Fiedler value λ₂(S, β). |
| [wsn/cost.py](wsn/cost.py) | Terrain and distance cost, the smoothed cost (η), and the k_max search. |
| [wsn/evaluation.py](wsn/evaluation.py) | `evaluate(problem, S)`: the original C, K_total, K, J, N_connected, unused budget, and bound/budget/connectivity checks; `budget_status` (active / near / inactive / violated). It is the single source of truth for results. |
| [wsn/pso.py](wsn/pso.py) | PSO with penalty fitness, φ = N − N_connected. It tracks the best feasible layout separately from gbest; the best feasible layout is the returned solution. |
| [wsn/sqp.py](wsn/sqp.py) | SLSQP continuation. Records status, iterations, evaluation counts, λ₂, smoothed-constraint status, and an original-model evaluation for every stage. Returns the best feasible stage, with the final stage kept separately. |
| [wsn/experiment.py](wsn/experiment.py) | Builds the experiment grid, runs (optionally in parallel), aggregates, and writes CSVs. |
| [wsn/visualize.py](wsn/visualize.py) | All report figures. |
| [main.py](main.py) | Command-line interface. |
| [scripts/verify_results.py](scripts/verify_results.py) | Independent verification of every saved layout. |
| [tests/test_sanity.py](tests/test_sanity.py) | Basic checks: grid, no double counting, soft→hard as β grows, BFS/λ₂ behaviour, cost model, solver smoke test. |

## Outputs (`results/`)

`results/` holds all four experiments (baseline, N sweep, α sweep, budget
sweep), produced by `python main.py --experiments all --workers 8`.

| File | Contents |
| --- | --- |
| `config.json` | Exact parameters, k_max, and experiments used for the run |
| `runs.csv` | One row per (method, N, α, b, seed). Reported solution (original model): `feasible`, `violations`, coverage, cost_total, budget, `unused_budget`, `budget_status`, cost_norm, objective (= original J), n_connected, `n_disconnected`. Also `found_feasible`, time_sec, cpu_sec, iteration and evaluation counts, solver success/status, plus method-specific diagnostics (see below) |
| `summary.csv` | Per (method, N, α, b): feasibility rate; counts of connectivity, budget, and bound violations; budget active/near counts; mean/sd/min/max of J, C, K, K_total, and unused budget **over feasible runs only** (`*_feas`); J over all runs (`objective_all_*`); time; evaluations. SQP rows also have the selected-β counts and how many final stages were feasible under the original model and under their own smoothed constraints |
| `sqp_stages.csv` | Per SQP continuation stage: β, `selected`, SLSQP status/message, iterations, evaluations, final λ₂, smoothed cost, `smooth_feasible` (stage met its own smoothed constraints), and the original-model C/K/J/feasibility/budget status of the stage solution |
| `history.csv` | Convergence. PSO: per iteration, gbest fitness F, gbest J, best feasible J, and feasible fraction of the swarm. SQP: per SLSQP iteration, the smoothed objective, original J, and original feasibility. |
| `positions.csv` | Sensor coordinates. `solution = reported` for every run; `solution = sqp_final_stage` for the SQP final-stage layout |
| `figures/cost_map.png` | Synthetic deployment cost map, with k_max and the base station |
| `figures/layouts_baseline_best.png`, `layouts_baseline_seed0.png`, `layouts_N*_best.png` | Placements: covered points, sensing disks, communication links (d ≤ r_c), base station, disconnected sensors marked ×, all over the cost map |
| `figures/sqp_selected_vs_final.png` | One SQP run where the reported (best feasible) stage and the infeasible final stage differ, side by side |
| `figures/convergence_baseline.png` | PSO per iteration (all seeds); SQP per SLSQP iteration for seed 0 with stage boundaries; SQP original J at each β stage for all seeds, with the reported stage ringed |
| `figures/baseline_comparison.png` | C, K, J, and time per seed for both methods (infeasible runs marked ×) |
| `figures/time_vs_N.png` | Time and objective evaluations vs N |
| `figures/metrics_vs_N.png`, `metrics_vs_alpha.png` | J, C, and K (feasible runs) plus feasibility rate vs N and vs α |
| `figures/tradeoff_alpha.png` | Coverage vs normalized cost per run, coloured by α |
| `figures/feasibility.png` | Feasible and infeasible counts with violation types per swept parameter |
| `figures/budget_sensitivity.png` | Coverage, J, and unused budget per run, plus feasibility rate, vs B = bN |

### Solution selection and penalties (revised)

- **SQP: best feasible continuation stage.**
  - **Evaluation:** after each β stage, the stage end-point is evaluated with the
    original model (hard coverage, exact cost, BFS connectivity with r_c = 1.8,
    bounds, budget).
  - **Reported solution:** the feasible end-point with the highest original J,
    recorded as `sqp_selected_stage` / `sqp_selected_beta`.
  - **Warm starts:** each stage still starts from the previous stage's result,
    even if that result is infeasible. Continuation stops early only if SLSQP
    returns a non-finite iterate (`sqp_aborted`).
  - **No feasible stage:** the final-stage layout is reported, with
    `found_feasible = False` and `feasible = False`.
  - **Final stage:** always kept separately: `sqp_final_*` columns, including
    the SLSQP flag and message (`sqp_final_success`, `sqp_final_message`),
    `sqp_final_smooth_feasible`, and the original-model `sqp_final_feasible`
    and `sqp_final_objective`. Its layout is in `positions.csv`.
  - **Smoothed λ₂ constraint:** satisfying it is never treated as proof of
    connectivity. Only the original-model check selects a stage.
- **PSO: unnormalized connectivity penalty.** φ(S) = N − N_connected(S), so each
  stranded sensor costs γ = 2 regardless of N. J lies in a range of width 1, so
  with γ > 1 any feasible layout outscores any disconnected one. The same F is
  used for particle evaluation and for personal and global best updates.

### What the reported numbers mean

- **PSO solution:** the best **feasible** layout seen in any evaluation. If none
  was ever feasible, the layout is the best-fitness one, and the run is marked
  infeasible with `found_feasible = False`. `pso_gbest_fitness` (penalized F),
  `pso_gbest_objective` (original J), `pso_gbest_feasible`, and
  `pso_gbest_n_disconnected` describe the best-fitness particle, which may differ
  from the reported layout.
- **SQP solution:** the best feasible stage, as described above. `solver_success`
  is SLSQP's flag for the **final** stage only, so a run can report a feasible
  solution while its final stage stopped at the iteration limit.
- **Statistics:** `summary.csv` statistics with the `_feas` suffix use only
  feasible runs, so infeasible layouts are never averaged into reported quality.

## Numerical-stability measures

- **Sigmoid and noisy-OR:** `log(1 − σ(z)) = log_expit(−z)` is evaluated
  directly. The product over sensors becomes a sum of logs, and `1 − exp(·)`
  uses `expm1`. This avoids both cancellation when σ ≈ 1 and underflow at large N.
- **Coincident nodes:** smooth functions use `sqrt(‖·‖² + 1e-12)`, so distances
  stay differentiable when two sensors (or a sensor and the base station or a
  grid point) coincide. Hard functions use exact distances.
- **Eigenvalues:** the Laplacian is explicitly symmetrized and `eigvalsh` is used.
  λ₂ is not differentiable where it is a repeated eigenvalue, for example when
  the graph splits into components and λ₁ = λ₂ = 0.
- **Bounds:** SLSQP iterates are clipped to M after each stage; the correction is
  at most about 1e-12.
