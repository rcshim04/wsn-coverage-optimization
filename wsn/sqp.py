"""Sequential Quadratic Programming with continuation (Sections 2.7 and 4.3).

For each beta in the schedule, SciPy's SLSQP solves

    min_S  J~(S, beta) = (1 - alpha) K~(S) - alpha C_bar(S, beta)
    s.t.   lambda_2(S, beta) - eps >= 0       (smooth connectivity, r_c_eff)
           B - K~_total(S)       >= 0          (smoothed budget)
           x_min <= x_i <= x_max, y_min <= y_i <= y_max

with finite-difference gradients, warm-starting each stage from the previous
stage's solution (even an infeasible one).

Solution selection: every stage end-point is scored with the ORIGINAL model
(hard coverage, exact cost, BFS connectivity with r_c, bounds, budget). The
reported solution is the feasible end-point with the highest original J; the
final stage's layout and SLSQP status are kept separately for diagnostics.
Satisfying the smoothed lambda_2 constraint is never taken as proof of
connectivity: only the original-model check counts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import minimize

from .cost import total_cost
from .evaluation import Evaluation, evaluate
from .geometry import algebraic_connectivity, soft_coverage_ratio
from .problem import Problem

SMOOTH_CON_TOL = 1e-6  # slack when reporting whether a stage met its own smoothed constraints


@dataclass
class StageInfo:
    beta: float
    success: bool
    status: int
    message: str
    n_iterations: int
    n_obj_evals: int
    n_con_evals: int
    smooth_objective: float
    lambda2: float  # lambda_2(S, beta) at stage end (constraint needs >= eps)
    smooth_cost_total: float
    smooth_feasible: bool  # stage end satisfies ITS OWN smoothed constraints
    eval: Evaluation  # original-model evaluation of the stage solution
    layout: np.ndarray


@dataclass
class SQPResult:
    """``layout`` is the REPORTED solution: the feasible stage end-point with the
    highest original J. If no stage was feasible it is the final stage's layout
    and ``found_feasible`` is False. The final stage is kept separately."""

    layout: np.ndarray
    found_feasible: bool
    selected_stage: Optional[int]  # index into stages; None if nothing feasible
    final_layout: np.ndarray
    x0: np.ndarray
    time_sec: float
    stages: List[StageInfo]
    n_obj_evals: int
    n_con_evals: int
    n_iterations: int
    aborted: str = ""  # non-empty if continuation stopped on a numerical failure
    # Per-SLSQP-iteration history across all stages (concatenated).
    hist_stage: List[int] = field(default_factory=list)
    hist_smooth_objective: List[float] = field(default_factory=list)
    hist_objective: List[float] = field(default_factory=list)  # original J
    hist_feasible: List[bool] = field(default_factory=list)

    @property
    def final(self) -> StageInfo:
        return self.stages[-1]

    @property
    def selected(self) -> Optional[StageInfo]:
        return None if self.selected_stage is None else self.stages[self.selected_stage]

    @property
    def success(self) -> bool:
        """SLSQP's own success flag for the final continuation stage."""
        return self.final.success

    @property
    def status(self) -> str:
        s = "; ".join(f"b={st.beta:g}:{st.message}" for st in self.stages)
        return s + (f"; ABORTED: {self.aborted}" if self.aborted else "")


def smooth_objective(problem: Problem, S: np.ndarray, beta: float) -> float:
    cfg = problem.config
    K = total_cost(S, cfg, smooth=True) / (problem.n_sensors * problem.k_max)
    C = soft_coverage_ratio(S, problem.points, cfg.sensing_radius, beta)
    return (1.0 - cfg.alpha) * K - cfg.alpha * C


def run_sqp(problem: Problem, seed: int, x0: Optional[np.ndarray] = None) -> SQPResult:
    cfg = problem.config
    n = problem.n_sensors
    if x0 is None:
        x0 = problem.random_layout(np.random.default_rng(seed))
    x = np.asarray(x0, float).reshape(-1).copy()
    bounds = list(zip(problem.lower, problem.upper))
    base, rc_eff, eps = problem.base_station, cfg.comm_radius_eff, cfg.lambda2_epsilon
    budget = cfg.budget

    counters = {"obj": 0, "con": 0}
    hist_stage, hist_sJ, hist_J, hist_feas = [], [], [], []
    stages: List[StageInfo] = []

    aborted = ""
    t0 = time.perf_counter()
    for k, beta in enumerate(cfg.beta_schedule):
        def fun(v, beta=beta):
            counters["obj"] += 1
            return smooth_objective(problem, v.reshape(n, 2), beta)

        def con_conn(v, beta=beta):
            counters["con"] += 1
            return algebraic_connectivity(v.reshape(n, 2), base, rc_eff, beta) - eps

        def con_budget(v):
            counters["con"] += 1
            return budget - total_cost(v.reshape(n, 2), cfg, smooth=True)

        def record(v, beta=beta, k=k):
            # Logging only: does not count toward the solver's evaluations.
            ev = evaluate(problem, v)
            hist_stage.append(k)
            hist_sJ.append(smooth_objective(problem, v.reshape(n, 2), beta))
            hist_J.append(ev.objective)
            hist_feas.append(ev.feasible)

        obj0, con0 = counters["obj"], counters["con"]
        record(x)
        res = minimize(
            fun, x, method="SLSQP", bounds=bounds,
            constraints=[{"type": "ineq", "fun": con_conn},
                         {"type": "ineq", "fun": con_budget}],
            options={"maxiter": cfg.sqp_max_iter, "ftol": cfg.sqp_ftol},
            callback=record,
        )
        if not np.all(np.isfinite(res.x)):
            # Numerical failure: continuing from a non-finite point is impossible.
            aborted = f"non-finite iterate at beta={beta:g} ({res.message})"
            break
        # SLSQP can step marginally outside bounds; project back (<= ~1e-12).
        x = np.clip(res.x, problem.lower, problem.upper)
        S = x.reshape(n, 2)
        lam2 = algebraic_connectivity(S, base, rc_eff, beta)
        smooth_cost = total_cost(S, cfg, smooth=True)
        stages.append(StageInfo(
            beta=float(beta), success=bool(res.success), status=int(res.status),
            message=str(res.message), n_iterations=int(res.nit),
            n_obj_evals=counters["obj"] - obj0, n_con_evals=counters["con"] - con0,
            smooth_objective=float(res.fun),
            lambda2=lam2,
            smooth_cost_total=smooth_cost,
            smooth_feasible=bool(lam2 - eps >= -SMOOTH_CON_TOL
                                 and budget - smooth_cost >= -SMOOTH_CON_TOL),
            eval=evaluate(problem, S),  # ORIGINAL model; this decides feasibility
            layout=S.copy(),
        ))
        # The next stage always warm-starts from this stage's result, feasible or not.

    if not stages:
        raise RuntimeError(f"SQP produced no usable stage: {aborted}")

    # Report the feasible (original model) stage end-point with the highest original J.
    feasible = [k for k, st in enumerate(stages) if st.eval.feasible]
    selected = max(feasible, key=lambda k: stages[k].eval.objective) if feasible else None
    final_layout = stages[-1].layout

    return SQPResult(
        layout=(stages[selected].layout if selected is not None else final_layout),
        found_feasible=selected is not None,
        selected_stage=selected,
        final_layout=final_layout,
        aborted=aborted,
        x0=np.asarray(x0, float).reshape(n, 2),
        time_sec=time.perf_counter() - t0,
        stages=stages,
        n_obj_evals=counters["obj"],
        n_con_evals=counters["con"],
        n_iterations=sum(s.n_iterations for s in stages),
        hist_stage=hist_stage,
        hist_smooth_objective=hist_sJ,
        hist_objective=hist_J,
        hist_feasible=hist_feas,
    )
