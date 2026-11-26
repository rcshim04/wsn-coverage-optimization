"""Problem definition: region M, sample grid P, base station, and k_max.

Solvers receive a :class:`Problem`; the sensor layout S (shape (N, 2)) is the
decision variable and is passed to evaluation functions separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import Config
from .cost import compute_k_max


@dataclass
class Problem:
    config: Config
    points: np.ndarray  # (|P|, 2) sample grid P
    base_station: np.ndarray  # (2,)
    k_max: float  # max single-sensor cost over M (fixed by the cost map)

    @property
    def n_sensors(self) -> int:
        return self.config.n_sensors

    @property
    def lower(self) -> np.ndarray:
        """Lower bounds of the flattened vector [x1, y1, ..., xN, yN]."""
        x_min, y_min, _, _ = self.config.region
        return np.tile([x_min, y_min], self.n_sensors).astype(float)

    @property
    def upper(self) -> np.ndarray:
        _, _, x_max, y_max = self.config.region
        return np.tile([x_max, y_max], self.n_sensors).astype(float)

    def random_layout(self, rng: np.random.Generator) -> np.ndarray:
        """Uniform random layout inside M, shape (N, 2)."""
        return rng.uniform(self.lower, self.upper).reshape(self.n_sensors, 2)


@lru_cache(maxsize=None)
def _k_max_cached(config_key: Config) -> float:
    return compute_k_max(config_key)


def build_problem(config: Config) -> Problem:
    """Discretize M into the uniform grid P (boundaries included)."""
    config.validate()
    x_min, y_min, x_max, y_max = config.region
    res = config.grid_resolution
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, res), np.linspace(y_min, y_max, res))
    points = np.column_stack([gx.ravel(), gy.ravel()])
    # k_max depends only on the cost map, not on N or alpha -> cache by those fields.
    cost_key = config.with_overrides(n_sensors=1, alpha=0.0, budget_per_sensor=0.0, seeds=(),
                                     output_dir="")
    return Problem(
        config=config,
        points=points,
        base_station=np.asarray(config.base_station, float),
        k_max=_k_max_cached(cost_key),
    )
