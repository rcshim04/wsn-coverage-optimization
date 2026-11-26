"""Independently re-verify every saved layout against the original model.

Reads <results>/positions.csv and runs.csv and recomputes coverage, cost,
bounds, budget, connectivity and J with code that does NOT reuse
wsn.evaluation (connectivity via scipy connected_components instead of the
package's BFS). Checks both the reported solution of every run and, for SQP,
the final continuation stage's layout. Then prints per-configuration
feasibility / objective / runtime statistics over the seeds.

    python scripts/verify_results.py [results_dir]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wsn.experiment import read_csv  # noqa: E402  (CSV loading only)

out = sys.argv[1] if len(sys.argv) > 1 else "results"
cfg = json.load(open(os.path.join(out, "config.json")))
runs = read_csv(os.path.join(out, "runs.csv"))
pos = read_csv(os.path.join(out, "positions.csv"))

x0, y0, x1, y1 = cfg["region"]
g = np.linspace(x0, x1, cfg["grid_resolution"])
P = np.array([(x, y) for y in g for x in g])
b = np.array(cfg["base_station"])
rs, rc, kmax = cfg["sensing_radius"], cfg["comm_radius"], cfg["k_max"]
assert rc == 1.8 and (x0, y0, x1, y1) == (0.0, 0.0, 10.0, 10.0)


def cost(S):
    t = sum(a * np.exp(-((S[:, 0] - mx) ** 2 + (S[:, 1] - my) ** 2) / (2 * s * s))
            for mx, my, a, s in cfg["terrain_regions"])
    return cfg["k0"] + cfg["w_d"] * cfg["kappa_d"] * np.linalg.norm(S - b, axis=1) + cfg["w_t"] * t


def recompute(S, alpha, bps):
    n = len(S)
    C = np.mean((cdist(S, P) <= rs).any(axis=0))
    Kt = cost(S).sum()
    J = alpha * C - (1 - alpha) * Kt / (n * kmax)
    nodes = np.vstack([S, b])
    _, lab = connected_components(cdist(nodes, nodes) <= rc, directed=False)
    nconn = int(np.sum(lab[:n] == lab[n]))
    inb = bool(np.all((S >= [x0 - 1e-9, y0 - 1e-9]) & (S <= [x1 + 1e-9, y1 + 1e-9])))
    B = bps * n
    feas = nconn == n and inb and Kt <= B + 1e-8
    return dict(C=C, Kt=Kt, J=J, nconn=nconn, inb=inb, unused=B - Kt, feas=feas)


layouts = {}
for p in pos:
    k = (p["method"], p["N"], float(p["alpha"]), float(p["budget_per_sensor"]), p["seed"], p["solution"])
    layouts.setdefault(k, []).append((p["sensor"], p["x"], p["y"]))

checked = mismatch = 0
for r in runs:
    key = (r["method"], r["N"], float(r["alpha"]), float(r["budget_per_sensor"]), r["seed"])
    sols = [("reported", "")] + ([("sqp_final_stage", "sqp_final_")] if r["method"] == "SQP" else [])
    for sol, pre in sols:
        S = np.array([xy for _, *xy in sorted(layouts[key + (sol,)])])
        v = recompute(S, r["alpha"], r["budget_per_sensor"])
        checks = {
            "coverage": abs(v["C"] - r[pre + "coverage"]) < 1e-12,
            "cost": abs(v["Kt"] - r[pre + "cost_total"]) < 1e-9,
            "objective=J": abs(v["J"] - r[pre + "objective"]) < 1e-9,
            "n_connected": v["nconn"] == r[pre + "n_connected"],
            "n_disconnected": r["N"] - v["nconn"] == r[pre + "n_disconnected"],
            "unused_budget": abs(v["unused"] - r[pre + "unused_budget"]) < 1e-9,
            "feasible": v["feas"] == r[pre + "feasible"],
        }
        if sol == "reported":
            checks["bounds"] = v["inb"] == r["within_bounds"]
            checks["found_feasible"] = r["found_feasible"] == r["feasible"]
        checked += 1
        if not all(checks.values()):
            mismatch += 1
            print("MISMATCH", key, sol, [k for k, ok in checks.items() if not ok])

print(f"re-verified {checked} layouts ({len(runs)} reported + SQP final stages): {mismatch} mismatches")
print(f"all reported layouts inside M: {all(r['within_bounds'] for r in runs)}")
nofeas = [r for r in runs if not r["found_feasible"]]
print(f"runs with NO feasible solution found: {len(nofeas)}"
      + "".join(f"\n  {r['method']} N={r['N']} a={r['alpha']} b={r['budget_per_sensor']} seed={r['seed']}: "
                f"{r['violations']}, {r['n_disconnected']} disconnected" for r in nofeas))

print("\nPer configuration over seeds (J stats over feasible runs; time over all runs):")
print(f"{'method':<5}{'N':>4}{'alpha':>6}{'b':>5}{'feas':>6}{'J mean':>9}{'J sd':>8}{'J min':>8}{'J max':>8}"
      f"{'time mean':>11}{'time sd':>9}")
groups = {}
for r in runs:
    groups.setdefault((r["method"], r["N"], r["alpha"], r["budget_per_sensor"]), []).append(r)
for (m, n, a, bb), rr in sorted(groups.items()):
    J = np.array([r["objective"] for r in rr if r["feasible"]])
    t = np.array([r["time_sec"] for r in rr])
    js = (f"{J.mean():>+9.4f}{J.std(ddof=1) if len(J) > 1 else 0:>8.4f}{J.min():>+8.3f}{J.max():>+8.3f}"
          if len(J) else f"{'-':>9}{'-':>8}{'-':>8}{'-':>8}")
    print(f"{m:<5}{n:>4}{a:>6g}{bb:>5g}{len(J):>4}/{len(rr)}{js}{t.mean():>11.2f}{t.std(ddof=1):>9.2f}")
