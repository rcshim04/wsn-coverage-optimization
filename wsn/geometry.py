"""Vectorized geometry: distances, coverage (hard + smooth), and connectivity.

Hard ("original") functions are used by PSO and for all final validation.
Smooth functions are only used inside SQP (Section 2.7).

Conventions: ``sensors`` has shape (N, 2), ``points`` has shape (P, 2).
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.special import expit, log_expit

# Tiny floor inside sqrt for the SMOOTH functions only. It keeps d(s, p)
# differentiable when two nodes coincide (||.|| has no gradient at 0), so
# SLSQP's finite differences never straddle a kink at d = 0. 1e-12 under the
# sqrt shifts distances by at most 1e-6, far below the grid spacing (0.256).
_DIST_FLOOR_SQ = 1e-12


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact Euclidean distances, shape (len(a), len(b))."""
    diff = np.asarray(a, float)[:, None, :] - np.asarray(b, float)[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def smooth_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """sqrt(||a - b||^2 + tiny): differentiable everywhere (SQP only)."""
    diff = np.asarray(a, float)[:, None, :] - np.asarray(b, float)[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff) + _DIST_FLOOR_SQ)


# ---------------------------------------------------------------------------
# Coverage (Section 2.2 hard, Section 2.7 smooth)
# ---------------------------------------------------------------------------
def coverage_mask(sensors: np.ndarray, points: np.ndarray, rs: float) -> np.ndarray:
    """(P,) bool: point covered by at least one sensor (binary disk model)."""
    return np.any(pairwise_distances(sensors, points) <= rs, axis=0)


def coverage_ratio(sensors: np.ndarray, points: np.ndarray, rs: float) -> float:
    """C(S) = (1/|P|) sum_p max_i 1[||s_i - p|| <= r_s]."""
    return float(np.mean(coverage_mask(sensors, points, rs)))


def soft_coverage_ratio(
    sensors: np.ndarray, points: np.ndarray, rs: float, beta: float
) -> float:
    """Smooth coverage C_bar(S, beta) with sigmoid indicators and noisy-OR.

        g_ip = sigma(beta (r_s - ||s_i - p||))
        C_bar = (1/|P|) sum_p [1 - prod_i (1 - g_ip)]

    Stability: 1 - sigma(z) = sigma(-z), so log(1 - g_ip) = log_expit(-z)
    exactly, without the cancellation of computing 1 - g when g ~ 1. The
    product over sensors is then a sum of logs (no underflow for large N).
    """
    z = beta * (rs - smooth_distances(sensors, points))  # (N, P)
    log_not_covered = np.sum(log_expit(-z), axis=0)  # log prod_i (1 - g_ip)
    return float(np.mean(-np.expm1(log_not_covered)))  # 1 - exp(.)


# ---------------------------------------------------------------------------
# Connectivity (Section 2.3 hard, Section 2.7 smooth)
# ---------------------------------------------------------------------------
def all_nodes(sensors: np.ndarray, base_station: np.ndarray) -> np.ndarray:
    """(N+1, 2) node array; the base station is the LAST node (index N)."""
    return np.vstack([np.asarray(sensors, float), np.reshape(base_station, (1, 2))])


def comm_adjacency(sensors, base_station, rc: float) -> np.ndarray:
    """Boolean adjacency of G(S, r_c): edge iff ||u - v|| <= r_c, u != v."""
    nodes = all_nodes(sensors, base_station)
    adj = pairwise_distances(nodes, nodes) <= rc
    np.fill_diagonal(adj, False)
    return adj


def comm_edges(sensors, base_station, rc: float):
    """List of (i, j), i < j, edges of G(S, r_c); index N is the base station."""
    adj = comm_adjacency(sensors, base_station, rc)
    iu, ju = np.nonzero(np.triu(adj, k=1))
    return list(zip(iu.tolist(), ju.tolist()))


def reachable_from_base(sensors, base_station, rc: float) -> np.ndarray:
    """(N,) bool: sensor reachable from b in G(S, r_c), by breadth-first search."""
    adj = comm_adjacency(sensors, base_station, rc)
    n = adj.shape[0] - 1
    seen = np.zeros(n + 1, dtype=bool)
    seen[n] = True
    queue = deque([n])
    while queue:
        u = queue.popleft()
        for v in np.flatnonzero(adj[u] & ~seen):
            seen[v] = True
            queue.append(v)
    return seen[:n]


def n_connected(sensors, base_station, rc: float) -> int:
    """N_connected(S): number of sensors with a (multi-hop) path to b."""
    return int(np.sum(reachable_from_base(sensors, base_station, rc)))


def is_connected(sensors, base_station, rc: float) -> bool:
    """True iff every sensor reaches the base station (G(S, r_c) connected)."""
    return bool(np.all(reachable_from_base(sensors, base_station, rc)))


def algebraic_connectivity(sensors, base_station, rc_eff: float, beta: float) -> float:
    """Smooth Fiedler value lambda_2(S, beta) of the sigmoid-weighted graph.

    w_ij = sigma(beta (r_c_eff - d_ij)), w_ii = 0, L = D - W, over the N
    sensors plus the base station. ``eigvalsh`` exploits symmetry (real,
    sorted eigenvalues). Note: lambda_2 is only differentiable where it is a
    simple eigenvalue; near multiplicity the finite-difference gradient can be
    inaccurate, which SLSQP tolerates but can slow convergence.
    """
    nodes = all_nodes(sensors, base_station)
    w = expit(beta * (rc_eff - smooth_distances(nodes, nodes)))
    np.fill_diagonal(w, 0.0)
    w = 0.5 * (w + w.T)  # enforce exact symmetry
    laplacian = np.diag(w.sum(axis=1)) - w
    return float(np.linalg.eigvalsh(laplacian)[1])
