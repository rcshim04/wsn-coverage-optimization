"""Location-dependent deployment cost (Sections 2.5 and 2.7).

    k(s)   = k0 + w_d k_d(s) + w_t k_t(s)
    k_d(s) = kappa_d ||s - b||                       (original)
    k~_d(s)= kappa_d sqrt(||s - b||^2 + eta^2)       (smoothed, SQP only)
    k_t(s) = sum_j a_j exp(-||s - mu_j||^2 / (2 sigma_j^2))
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .config import Config


def terrain_cost(points: np.ndarray, config: Config) -> np.ndarray:
    """k_t at each row of ``points`` (shape (M, 2)) -> (M,)."""
    pts = np.asarray(points, float)
    total = np.zeros(len(pts))
    for mx, my, a, sigma in config.terrain_regions:
        sq = (pts[:, 0] - mx) ** 2 + (pts[:, 1] - my) ** 2
        total += a * np.exp(-sq / (2.0 * sigma**2))
    return total


def sensor_costs(points: np.ndarray, config: Config, smooth: bool = False) -> np.ndarray:
    """Per-sensor installation cost k(s_i) (or the smoothed k~ if ``smooth``)."""
    pts = np.asarray(points, float)
    sq = np.sum((pts - np.asarray(config.base_station)) ** 2, axis=1)
    dist = np.sqrt(sq + config.eta**2) if smooth else np.sqrt(sq)
    return config.k0 + config.w_d * config.kappa_d * dist + config.w_t * terrain_cost(pts, config)


def total_cost(sensors: np.ndarray, config: Config, smooth: bool = False) -> float:
    """K_total(S) = sum_i k(s_i)."""
    return float(np.sum(sensor_costs(sensors, config, smooth)))


def compute_k_max(config: Config, grid: int = 401) -> float:
    """k_max = max_{s in M} k(s), from a dense grid search refined locally.

    The grid locates the global basin (k is multimodal: three Gaussian bumps
    plus a cone); bounded L-BFGS-B polishes the best few grid points.
    """
    x_min, y_min, x_max, y_max = config.region
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, grid), np.linspace(y_min, y_max, grid))
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    vals = sensor_costs(pts, config)
    best = float(vals.max())
    bounds = [(x_min, x_max), (y_min, y_max)]
    for idx in np.argsort(vals)[-5:]:
        res = minimize(lambda s: -sensor_costs(s[None, :], config)[0], pts[idx],
                       method="L-BFGS-B", bounds=bounds)
        best = max(best, -float(res.fun))
    return best
