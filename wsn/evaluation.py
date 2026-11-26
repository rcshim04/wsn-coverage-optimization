"""Original (non-smoothed) objective and constraint evaluation (Sections 2.2-2.6).

Every final solution - from PSO or SQP - is scored here, so both algorithms
are judged by exactly the same functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cost import total_cost
from .geometry import coverage_ratio, n_connected
from .problem import Problem

BOUNDARY_TOL = 1e-9  # numerical slack for "s_i in M"
BUDGET_TOL = 1e-8  # absolute slack on K_total <= B (SLSQP may sit on the bound)


@dataclass
class Evaluation:
    coverage: float  # C(S)
    cost_total: float  # K_total(S)
    cost_norm: float  # K(S) = K_total / (N k_max)
    objective: float  # J(S) = alpha C - (1 - alpha) K
    n_connected: int  # sensors reachable from b in G(S, r_c)
    connected: bool
    within_bounds: bool
    within_budget: bool
    budget: float
    n_sensors: int

    @property
    def feasible(self) -> bool:
        return self.connected and self.within_bounds and self.within_budget

    @property
    def unused_budget(self) -> float:
        """B - K_total (negative when the budget is violated)."""
        return self.budget - self.cost_total

    @property
    def n_disconnected(self) -> int:
        return self.n_sensors - self.n_connected

    @property
    def violations(self) -> str:
        v = [name for name, ok in (("connectivity", self.connected),
                                   ("boundary", self.within_bounds),
                                   ("budget", self.within_budget)) if not ok]
        return "+".join(v) if v else "none"


def objective(coverage: float, cost_norm: float, alpha: float) -> float:
    return alpha * coverage - (1.0 - alpha) * cost_norm


def evaluate(problem: Problem, sensors: np.ndarray) -> Evaluation:
    cfg = problem.config
    S = np.asarray(sensors, float).reshape(-1, 2)
    n = S.shape[0]
    C = coverage_ratio(S, problem.points, cfg.sensing_radius)
    K_tot = total_cost(S, cfg)
    K = K_tot / (n * problem.k_max)
    n_conn = n_connected(S, problem.base_station, cfg.comm_radius)
    x_min, y_min, x_max, y_max = cfg.region
    in_bounds = bool(
        np.all(S[:, 0] >= x_min - BOUNDARY_TOL) and np.all(S[:, 0] <= x_max + BOUNDARY_TOL)
        and np.all(S[:, 1] >= y_min - BOUNDARY_TOL) and np.all(S[:, 1] <= y_max + BOUNDARY_TOL)
    )
    budget = cfg.budget_per_sensor * n
    return Evaluation(
        coverage=C,
        cost_total=K_tot,
        cost_norm=K,
        objective=objective(C, K, cfg.alpha),
        n_connected=n_conn,
        connected=n_conn == n,
        within_bounds=in_bounds,
        within_budget=K_tot <= budget + BUDGET_TOL,
        budget=budget,
        n_sensors=n,
    )


def budget_status(ev: Evaluation, cfg) -> str:
    """'violated', 'active' (on the bound), 'near' (within budget_near_tol of B), or 'inactive'."""
    rel = ev.unused_budget / ev.budget
    if not ev.within_budget:
        return "violated"
    if rel <= cfg.budget_active_tol:
        return "active"
    if rel <= cfg.budget_near_tol:
        return "near"
    return "inactive"
