"""Basic sanity checks that the model functions behave as the report defines them.

Run with:  python -m pytest      (or)      python tests/test_sanity.py
"""

from __future__ import annotations

import numpy as np

from wsn.config import DEFAULT_CONFIG
from wsn.cost import sensor_costs, total_cost
from wsn.evaluation import evaluate
from wsn.geometry import (algebraic_connectivity, coverage_ratio, is_connected,
                          n_connected, soft_coverage_ratio)
from wsn.problem import build_problem
from wsn.pso import run_pso
from wsn.sqp import run_sqp

CFG = DEFAULT_CONFIG
P = build_problem(CFG)
B = P.base_station


def test_grid_and_parameters():
    assert len(P.points) == 1600
    assert np.isclose(P.points[:, 0].min(), 0) and np.isclose(P.points[:, 0].max(), 10)
    assert np.isclose(np.diff(np.unique(P.points[:, 0]))[0], 10 / 39)
    assert CFG.sensing_radius == CFG.comm_radius == 1.8 and np.isclose(CFG.comm_radius_eff, 1.4)
    assert np.isclose(CFG.budget, 18.0)


def test_hard_coverage_counts_points_once():
    one = np.array([[5.0, 5.0]])
    c1 = coverage_ratio(one, P.points, 1.8)
    expected = np.mean(np.linalg.norm(P.points - one, axis=1) <= 1.8)
    assert np.isclose(c1, expected)
    assert np.isclose(coverage_ratio(np.repeat(one, 3, 0), P.points, 1.8), c1)  # no double count


def test_soft_coverage_approaches_hard_and_is_stable():
    S = P.random_layout(np.random.default_rng(1))
    hard = coverage_ratio(S, P.points, 1.8)
    errs = [abs(soft_coverage_ratio(S, P.points, 1.8, b) - hard) for b in (1, 8, 50, 1000)]
    assert errs[-1] < 0.01 and errs[-1] < errs[0]
    # coincident sensors / huge beta: finite, within [0, 1]
    v = soft_coverage_ratio(np.zeros((5, 2)), P.points, 1.8, 1e4)
    assert np.isfinite(v) and 0 <= v <= 1


def test_connectivity_bfs_and_fiedler():
    chain = np.array([[5 + 1.7 * k, 5.0] for k in range(1, 3)])  # 1.7 spacing <= r_c
    assert is_connected(chain, B, 1.8)
    far = np.vstack([chain, [[0.5, 0.5]]])
    assert not is_connected(far, B, 1.8) and n_connected(far, B, 1.8) == 2
    near = np.array([[5.5, 5.0], [6.0, 5.0]])
    assert algebraic_connectivity(near, B, 1.4, 50) > CFG.lambda2_epsilon
    assert algebraic_connectivity(far, B, 1.4, 50) < 1e-6
    assert np.isfinite(algebraic_connectivity(np.repeat(B[None], 3, 0), B, 1.4, 50))


def test_cost_model():
    k_b = sensor_costs(B[None], CFG)[0]
    assert k_b > CFG.k0 and P.k_max >= sensor_costs(P.points, CFG).max() - 1e-12
    # Gaussian centre of region 1 costs more than its far surroundings
    assert sensor_costs(np.array([[2.5, 7.0]]), CFG)[0] > sensor_costs(np.array([[2.5, 9.9]]), CFG)[0]
    S = P.random_layout(np.random.default_rng(0))
    assert total_cost(S, CFG, smooth=True) >= total_cost(S, CFG)  # sqrt(d^2+eta^2) >= d
    ev = evaluate(P, S)
    assert 0 <= ev.cost_norm <= 1 and np.isclose(ev.objective, 0.7 * ev.coverage - 0.3 * ev.cost_norm)


def test_solvers_smoke():
    cfg = CFG.with_overrides(n_sensors=4, pso_swarm_size=10, pso_iterations=10,
                             beta_schedule=(1.0, 8.0), sqp_max_iter=20)
    prob = build_problem(cfg)
    r = run_pso(prob, seed=0)
    assert len(r.hist_gbest_fitness) == 11 and r.n_evaluations == 110
    assert evaluate(prob, r.layout).feasible == r.found_feasible
    s = run_sqp(prob, seed=0)
    assert len(s.stages) == 2 and evaluate(prob, s.layout).within_bounds
    # SQP reports the best FEASIBLE stage (original model), else the final stage flagged infeasible
    feas = [st.eval.objective for st in s.stages if st.eval.feasible]
    if feas:
        assert s.found_feasible and np.isclose(evaluate(prob, s.layout).objective, max(feas))
    else:
        assert not s.found_feasible and np.allclose(s.layout, s.final_layout)


def test_pso_penalty_counts_disconnected_sensors():
    from wsn.pso import penalized_fitness
    far = np.array([[5.5, 5.0], [0.5, 0.5], [9.5, 9.5]])  # two stranded sensors
    ev = evaluate(P, far)
    assert np.isclose(penalized_fitness(P, ev), ev.objective - CFG.pso_gamma * 2)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
