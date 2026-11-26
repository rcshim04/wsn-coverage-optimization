"""Central configuration for the WSN coverage-cost optimization project.

Every numerical parameter from Section 3 of the technical report lives here, so
experiments and sweeps override a single frozen dataclass instead of threading
arguments through every module. Use ``Config.with_overrides(...)`` to get a
modified copy (e.g. a different N or alpha).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

# Synthetic terrain cost map (report Section 3.1, table): (mu_x, mu_y, a_j, sigma_j)
TERRAIN_REGIONS: Tuple[Tuple[float, float, float, float], ...] = (
    (2.5, 7.0, 0.8, 1.2),
    (7.5, 7.0, 0.6, 1.5),
    (6.0, 2.5, 0.7, 1.3),
)


@dataclass(frozen=True)
class Config:
    # ----- Problem definition (Sections 3.1-3.2) -----------------------------
    n_sensors: int = 10
    """Number of sensors N (baseline N = 10). Decision vector has 2N entries."""

    region: Tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0)
    """Deployment region M as (x_min, y_min, x_max, y_max)."""

    base_station: Tuple[float, float] = (5.0, 5.0)
    """Fixed base station b (communication only, provides no coverage)."""

    sensing_radius: float = 1.8  # r_s
    comm_radius: float = 1.8  # r_c

    grid_resolution: int = 40
    """|P| = 40 x 40 = 1600 uniformly spaced points including the boundary."""

    # ----- Deployment cost (Sections 2.5, 3.1) -------------------------------
    k0: float = 1.0  # fixed installation cost per sensor
    kappa_d: float = 0.05  # distance cost per unit distance from b
    w_d: float = 1.0  # weight on distance cost
    w_t: float = 1.0  # weight on terrain cost
    terrain_regions: Tuple[Tuple[float, float, float, float], ...] = TERRAIN_REGIONS
    budget_per_sensor: float = 1.8
    """Budget B(N) = budget_per_sensor * N."""

    # ----- Objective (Section 2.6) -------------------------------------------
    alpha: float = 0.7
    """Coverage-cost weight: J = alpha*C - (1-alpha)*K (baseline 0.7)."""

    # ----- PSO (Sections 2.8, 3.3) -------------------------------------------
    pso_swarm_size: int = 40
    pso_iterations: int = 120
    pso_inertia: float = 0.7  # w
    pso_cognitive: float = 1.5  # c1
    pso_social: float = 1.5  # c2
    pso_velocity_clamp_frac: float = 0.3
    """|v| <= 0.3 * region width, per coordinate."""
    pso_gamma: float = 2.0  # connectivity penalty coefficient
    pso_rho: float = 2.0  # budget penalty coefficient

    # ----- SQP / smooth reformulation (Sections 2.7, 3.2, 3.3) ---------------
    beta_schedule: Tuple[float, ...] = (1.0, 3.0, 8.0, 20.0, 50.0)
    sqp_max_iter: int = 200  # per continuation stage
    sqp_ftol: float = 1e-7  # SLSQP stopping tolerance (not specified in report)
    lambda2_epsilon: float = 0.05  # epsilon in lambda_2(S, beta) >= epsilon
    comm_margin_delta: float = 0.4  # delta: r_c_eff = r_c - delta = 1.4
    eta: float = 0.01  # smoothing of ||s - b|| in the distance cost

    # ----- Experiments (Section 3.3) -----------------------------------------
    seeds: Tuple[int, ...] = tuple(range(8))
    n_values: Tuple[int, ...] = (5, 10, 15, 20, 30)
    alpha_values: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.7, 1.0)
    budget_values: Tuple[float, ...] = (1.4, 1.5, 1.6, 1.8)
    """Budget sensitivity: B(N) = b N for each b (at the baseline N and alpha)."""

    # Budget-activity classification of a solution (unused budget = B - K_total):
    budget_active_tol: float = 1e-6  # unused / B <= this  -> "active" (on the bound)
    budget_near_tol: float = 0.02  # unused / B <= this -> "near" (within 2% of B)

    output_dir: str = "results"

    def with_overrides(self, **kwargs) -> "Config":
        return replace(self, **kwargs)

    # ----- Derived helpers ---------------------------------------------------
    @property
    def region_width(self) -> float:
        return self.region[2] - self.region[0]

    @property
    def region_height(self) -> float:
        return self.region[3] - self.region[1]

    @property
    def n_points(self) -> int:
        return self.grid_resolution * self.grid_resolution

    @property
    def budget(self) -> float:
        return self.budget_per_sensor * self.n_sensors

    @property
    def comm_radius_eff(self) -> float:
        return self.comm_radius - self.comm_margin_delta

    def validate(self) -> None:
        x_min, y_min, x_max, y_max = self.region
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("region must have positive width and height")
        if self.n_sensors < 1:
            raise ValueError("n_sensors must be >= 1")
        if self.sensing_radius <= 0 or self.comm_radius <= 0:
            raise ValueError("radii must be positive")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.comm_radius_eff <= 0:
            raise ValueError("comm_margin_delta must be smaller than comm_radius")
        bx, by = self.base_station
        if not (x_min <= bx <= x_max and y_min <= by <= y_max):
            raise ValueError("base station must lie inside the region")


DEFAULT_CONFIG = Config()
