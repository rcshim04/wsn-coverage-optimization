"""Hand-written Particle Swarm Optimization (Sections 2.8 and 4.2).

Each particle is a full layout flattened to [x1, y1, ..., xN, yN] (2N dims).
Fitness uses the ORIGINAL coverage and cost functions plus penalties:

    F(S) = J(S) - gamma * phi(S) - rho * max(0, (K_total(S) - B) / B)
    phi(S) = N - N_connected(S)      (number of disconnected sensors)

Updates (standard inertia-weight PSO):
    v <- w v + c1 r1 (pbest - x) + c2 r2 (gbest - x),  |v_d| <= vmax
    x <- clip(x + v, M)

Because penalties do not guarantee feasibility, the best FEASIBLE layout seen
in any evaluation is stored separately from the best-fitness (gbest) layout.
The returned solution is the best feasible one; if none was ever found the
gbest layout is returned and flagged ``found_feasible = False``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .evaluation import Evaluation, evaluate
from .problem import Problem


@dataclass
class PSOResult:
    layout: np.ndarray  # final solution (best feasible, else best fitness)
    found_feasible: bool
    best_fitness: float  # F of gbest
    best_fitness_layout: np.ndarray
    best_fitness_eval: Evaluation
    n_evaluations: int
    n_iterations: int
    time_sec: float
    # Per-iteration histories (index 0 = initial swarm).
    hist_gbest_fitness: List[float] = field(default_factory=list)
    hist_gbest_objective: List[float] = field(default_factory=list)
    hist_best_feasible_objective: List[float] = field(default_factory=list)  # nan until found
    hist_feasible_fraction: List[float] = field(default_factory=list)
    status: str = ""


def connectivity_violation(ev: Evaluation) -> int:
    """phi(S) = N - N_connected(S): the number of sensors with no path to b.

    Not normalized by N, so each stranded sensor costs gamma regardless of N.
    Since J lies in [-(1 - alpha), alpha] (a range of width 1), gamma > 1
    makes any feasible layout outscore any disconnected one.
    """
    return ev.n_sensors - ev.n_connected


def penalized_fitness(problem: Problem, ev: Evaluation) -> float:
    cfg = problem.config
    phi = connectivity_violation(ev)
    budget_violation = max(0.0, (ev.cost_total - ev.budget) / ev.budget)
    return ev.objective - cfg.pso_gamma * phi - cfg.pso_rho * budget_violation


def run_pso(problem: Problem, seed: int, config=None) -> PSOResult:
    cfg = config or problem.config
    rng = np.random.default_rng(seed)
    n, n_part = problem.n_sensors, cfg.pso_swarm_size
    lower, upper = problem.lower, problem.upper
    vmax = cfg.pso_velocity_clamp_frac * cfg.region_width

    t0 = time.perf_counter()
    pos = rng.uniform(lower, upper, size=(n_part, 2 * n))
    vel = rng.uniform(-vmax, vmax, size=(n_part, 2 * n))

    n_evals = 0
    best_feas_J, best_feas_x = -np.inf, None

    def evaluate_swarm(X):
        nonlocal n_evals, best_feas_J, best_feas_x
        fits, evs = np.empty(len(X)), []
        for i, x in enumerate(X):
            ev = evaluate(problem, x)
            fits[i] = penalized_fitness(problem, ev)
            evs.append(ev)
            if ev.feasible and ev.objective > best_feas_J:
                best_feas_J, best_feas_x = ev.objective, x.copy()
        n_evals += len(X)
        return fits, evs

    fit, evs = evaluate_swarm(pos)
    pbest_x, pbest_f = pos.copy(), fit.copy()
    pbest_ev = list(evs)
    g = int(np.argmax(pbest_f))

    hist = {"F": [], "J": [], "feasJ": [], "feasfrac": []}

    def log(evs_now):
        hist["F"].append(float(pbest_f[g]))
        hist["J"].append(pbest_ev[g].objective)
        hist["feasJ"].append(best_feas_J if best_feas_x is not None else np.nan)
        hist["feasfrac"].append(float(np.mean([e.feasible for e in evs_now])))

    log(evs)
    w, c1, c2 = cfg.pso_inertia, cfg.pso_cognitive, cfg.pso_social
    for _ in range(cfg.pso_iterations):
        r1 = rng.random(pos.shape)
        r2 = rng.random(pos.shape)
        vel = w * vel + c1 * r1 * (pbest_x - pos) + c2 * r2 * (pbest_x[g] - pos)
        vel = np.clip(vel, -vmax, vmax)
        pos = np.clip(pos + vel, lower, upper)  # boundary constraint by clipping

        fit, evs = evaluate_swarm(pos)
        improved = fit > pbest_f
        pbest_x[improved] = pos[improved]
        pbest_f[improved] = fit[improved]
        for i in np.flatnonzero(improved):
            pbest_ev[i] = evs[i]
        g = int(np.argmax(pbest_f))
        log(evs)

    elapsed = time.perf_counter() - t0
    gbest_layout = pbest_x[g].reshape(n, 2)
    found = best_feas_x is not None
    return PSOResult(
        layout=(best_feas_x.reshape(n, 2) if found else gbest_layout),
        found_feasible=found,
        best_fitness=float(pbest_f[g]),
        best_fitness_layout=gbest_layout,
        best_fitness_eval=pbest_ev[g],
        n_evaluations=n_evals,
        n_iterations=cfg.pso_iterations,
        time_sec=elapsed,
        hist_gbest_fitness=hist["F"],
        hist_gbest_objective=hist["J"],
        hist_best_feasible_objective=hist["feasJ"],
        hist_feasible_fraction=hist["feasfrac"],
        status=("max iterations reached" if found
                else "max iterations reached; no feasible layout found"),
    )
