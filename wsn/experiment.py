"""Experiment runner (Section 3.3): baseline, N sweep, alpha sweep, budget sweep.

Each (N, alpha, b, method, seed) run is independent, so runs are distributed over
worker processes. Every reported layout is re-scored with the ORIGINAL model in
:func:`wsn.evaluation.evaluate`; feasibility means connectivity (BFS with r_c),
boundary, and budget B = b N are all satisfied.

Outputs (CSV, one file per table):
    runs.csv        one row per run: reported solution metrics, feasibility, time,
                    evaluations, PSO/SQP diagnostics (incl. SQP final stage)
    summary.csv     per (method, N, alpha, b): feasibility rate + statistics
    sqp_stages.csv  per SQP continuation stage: solver status, smoothed-constraint
                    status, original-model eval, and whether it was selected
    history.csv     convergence histories (per PSO iteration / SLSQP iteration)
    positions.csv   sensor coordinates: the reported solution of every run, plus
                    the SQP final-stage layout (column ``solution``)
"""

from __future__ import annotations

import csv
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .config import Config
from .evaluation import budget_status, evaluate
from .problem import build_problem
from .pso import run_pso
from .sqp import run_sqp

METHODS = ("PSO", "SQP")


def experiment_grid(cfg: Config, experiments: Iterable[str]) -> List[Tuple[int, float, float]]:
    """Unique (N, alpha, b) configurations needed by the requested experiments."""
    b0 = cfg.budget_per_sensor
    configs = []
    for name in experiments:
        if name == "baseline":
            configs.append((cfg.n_sensors, cfg.alpha, b0))
        elif name == "nsweep":
            configs += [(n, cfg.alpha, b0) for n in cfg.n_values]
        elif name == "alphasweep":
            configs += [(cfg.n_sensors, a, b0) for a in cfg.alpha_values]
        elif name == "budget":
            configs += [(cfg.n_sensors, cfg.alpha, b) for b in cfg.budget_values]
        else:
            raise ValueError(f"unknown experiment {name!r}")
    return list(dict.fromkeys(configs))  # dedupe, keep order


def _eval_cols(ev, cfg, prefix=""):
    return {
        f"{prefix}feasible": ev.feasible,
        f"{prefix}violations": ev.violations,
        f"{prefix}coverage": ev.coverage,
        f"{prefix}cost_total": ev.cost_total,
        f"{prefix}cost_norm": ev.cost_norm,
        f"{prefix}objective": ev.objective,
        f"{prefix}n_connected": ev.n_connected,
        f"{prefix}n_disconnected": ev.n_disconnected,
        f"{prefix}unused_budget": ev.unused_budget,
        f"{prefix}budget_status": budget_status(ev, cfg),
    }


def run_single(task) -> Dict:
    """Run one solver from one seed; return CSV-ready rows. (Top-level for pickling.)"""
    base_cfg, n, alpha, b, method, seed = task
    cfg = base_cfg.with_overrides(n_sensors=n, alpha=alpha, budget_per_sensor=b)
    problem = build_problem(cfg)
    key = {"method": method, "N": n, "alpha": alpha, "budget_per_sensor": b, "seed": seed}
    extra_positions = []

    cpu0 = time.process_time()
    if method == "PSO":
        res = run_pso(problem, seed)
        layout = res.layout
        g = res.best_fitness_eval
        extra = {
            "found_feasible": res.found_feasible,
            "n_iterations": res.n_iterations,
            "n_obj_evals": res.n_evaluations,
            "n_con_evals": res.n_evaluations,  # constraints checked in every evaluation
            "solver_success": res.found_feasible,
            "status": res.status,
            # Highest penalized-fitness particle (gbest) vs the reported solution:
            "pso_gbest_fitness": res.best_fitness,
            "pso_gbest_objective": g.objective,
            "pso_gbest_feasible": g.feasible,
            "pso_gbest_n_disconnected": g.n_disconnected,
        }
        history = [
            {**key, "step": i, "stage": "", "best_fitness": f, "objective": j,
             "best_feasible_objective": fj, "feasible_fraction": ff, "smooth_objective": "",
             "feasible": ""}
            for i, (f, j, fj, ff) in enumerate(zip(
                res.hist_gbest_fitness, res.hist_gbest_objective,
                res.hist_best_feasible_objective, res.hist_feasible_fraction))
        ]
        stages = []
    else:
        res = run_sqp(problem, seed)
        layout = res.layout
        fin, sel = res.final, res.selected
        extra = {
            "found_feasible": res.found_feasible,
            "n_iterations": res.n_iterations,
            "n_obj_evals": res.n_obj_evals,
            "n_con_evals": res.n_con_evals,
            "solver_success": res.success,  # SLSQP flag of the FINAL stage
            "status": res.status,
            "sqp_selected_stage": "" if sel is None else res.selected_stage,
            "sqp_selected_beta": "" if sel is None else sel.beta,
            "sqp_stages_converged": sum(s.success for s in res.stages),
            "sqp_n_stages_run": len(res.stages),
            "sqp_aborted": res.aborted,
            # Final continuation stage, kept separately for diagnostics:
            "sqp_final_beta": fin.beta,
            "sqp_final_success": fin.success,
            "sqp_final_message": fin.message,
            "sqp_final_lambda2": fin.lambda2,
            "sqp_final_smooth_feasible": fin.smooth_feasible,
            **_eval_cols(fin.eval, cfg, prefix="sqp_final_"),
        }
        history = [
            {**key, "step": i, "stage": st, "best_fitness": "", "objective": j,
             "best_feasible_objective": "", "feasible_fraction": "",
             "smooth_objective": sj, "feasible": f}
            for i, (st, sj, j, f) in enumerate(zip(
                res.hist_stage, res.hist_smooth_objective, res.hist_objective, res.hist_feasible))
        ]
        stages = [
            {**key, "stage": k, "beta": s.beta, "selected": k == res.selected_stage,
             "success": s.success, "status_code": s.status, "message": s.message,
             "n_iterations": s.n_iterations, "n_obj_evals": s.n_obj_evals,
             "n_con_evals": s.n_con_evals, "smooth_objective": s.smooth_objective,
             "lambda2": s.lambda2, "smooth_cost_total": s.smooth_cost_total,
             "smooth_feasible": s.smooth_feasible, **_eval_cols(s.eval, cfg)}
            for k, s in enumerate(res.stages)
        ]
        extra_positions = [{**key, "solution": "sqp_final_stage", "sensor": i, "x": x, "y": y}
                           for i, (x, y) in enumerate(res.final_layout)]
    cpu = time.process_time() - cpu0

    ev = evaluate(problem, layout)  # independent re-validation with original model
    run = {
        **key,
        **_eval_cols(ev, cfg),
        "budget": ev.budget,
        "connected": ev.connected,
        "within_bounds": ev.within_bounds,
        "within_budget": ev.within_budget,
        "time_sec": res.time_sec,
        "cpu_sec": cpu,
        **extra,
    }
    positions = [{**key, "solution": "reported", "sensor": i, "x": x, "y": y}
                 for i, (x, y) in enumerate(layout)] + extra_positions
    return {"run": run, "history": history, "stages": stages, "positions": positions}


def run_experiments(cfg: Config, experiments: Iterable[str], workers: int = 1,
                    progress: bool = True) -> Dict[str, List[Dict]]:
    grid = experiment_grid(cfg, experiments)
    tasks = [(cfg, n, a, b, m, s) for (n, a, b) in grid for m in METHODS for s in cfg.seeds]
    # Longest tasks first (SQP, large N) so the pool finishes evenly.
    tasks.sort(key=lambda t: (t[4] != "SQP", -t[1]))
    out = {"runs": [], "history": [], "stages": [], "positions": []}

    def collect(result, i):
        r = result["run"]
        out["runs"].append(r)
        out["history"] += result["history"]
        out["stages"] += result["stages"]
        out["positions"] += result["positions"]
        if progress:
            print(f"[{i:>3}/{len(tasks)}] {r['method']} N={r['N']:<2} a={r['alpha']:<4} "
                  f"b={r['budget_per_sensor']:<3} seed={r['seed']}  J={r['objective']:+.4f}  "
                  f"C={r['coverage']:.3f}  feasible={r['feasible']!s:<5}  t={r['time_sec']:.1f}s",
                  flush=True)

    if workers <= 1:
        for i, t in enumerate(tasks, 1):
            collect(run_single(t), i)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, result in enumerate(pool.map(run_single, tasks), 1):
                collect(result, i)

    order = lambda r: (r["method"], r["N"], r["alpha"], r["budget_per_sensor"], r["seed"])  # noqa: E731
    for k in out:
        out[k].sort(key=lambda r: (*order(r), r.get("solution", ""), r.get("step", 0),
                                   r.get("stage", 0), r.get("sensor", 0)))
    out["summary"] = summarize(out["runs"])
    return out


def _stats(values, prefix):
    v = np.asarray(values, float)
    if len(v) == 0:
        return {f"{prefix}_{s}": "" for s in ("mean", "std", "min", "max")}
    return {f"{prefix}_mean": v.mean(), f"{prefix}_std": v.std(ddof=1) if len(v) > 1 else 0.0,
            f"{prefix}_min": v.min(), f"{prefix}_max": v.max()}


def summarize(runs: List[Dict]) -> List[Dict]:
    """Per (method, N, alpha, b). Objective/coverage/cost stats use FEASIBLE runs only;
    ``*_all`` columns include infeasible runs and are labelled as such."""
    groups: Dict[tuple, List[Dict]] = {}
    for r in runs:
        groups.setdefault((r["method"], r["N"], r["alpha"], r.get("budget_per_sensor", 1.8)),
                          []).append(r)
    rows = []
    for (m, n, a, b), rs in sorted(groups.items()):
        feas = [r for r in rs if r["feasible"]]
        bs = Counter(r["budget_status"] for r in rs)
        row = {"method": m, "N": n, "alpha": a, "budget_per_sensor": b, "budget": rs[0]["budget"],
               "n_runs": len(rs), "n_feasible": len(feas),
               "feasibility_rate": len(feas) / len(rs),
               "n_conn_violations": sum(not r["connected"] for r in rs),
               "n_budget_violations": sum(not r["within_budget"] for r in rs),
               "n_bound_violations": sum(not r["within_bounds"] for r in rs),
               "n_budget_active": bs["active"], "n_budget_near": bs["near"],
               "n_solver_success": sum(bool(r["solver_success"]) for r in rs)}
        if m == "SQP":
            sel = Counter(r["sqp_selected_beta"] for r in rs if r["found_feasible"])
            row.update({
                "sqp_selected_beta_counts": " ".join(f"b{k:g}:{v}" for k, v in sorted(sel.items())),
                "sqp_n_final_stage_feasible": sum(bool(r["sqp_final_feasible"]) for r in rs),
                "sqp_n_final_stage_smooth_feasible": sum(bool(r["sqp_final_smooth_feasible"]) for r in rs),
            })
            row.update(_stats([r["sqp_final_objective"] for r in rs if r["sqp_final_feasible"]],
                              "sqp_final_objective_feas"))
        row.update(_stats([r["objective"] for r in feas], "objective_feas"))
        row.update(_stats([r["coverage"] for r in feas], "coverage_feas"))
        row.update(_stats([r["cost_norm"] for r in feas], "cost_norm_feas"))
        row.update(_stats([r["cost_total"] for r in feas], "cost_total_feas"))
        row.update(_stats([r["unused_budget"] for r in feas], "unused_budget_feas"))
        row.update(_stats([r["objective"] for r in rs], "objective_all"))
        row.update(_stats([r["time_sec"] for r in rs], "time_sec"))
        row.update(_stats([r["n_obj_evals"] for r in rs], "n_obj_evals"))
        rows.append(row)
    return rows


def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else repr(float(v))  # full precision (round-trips exactly)
    return v


def write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in fields})


def save_results(out: Dict[str, List[Dict]], out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name in ("runs", "summary", "sqp_stages", "history", "positions"):
        key = "stages" if name == "sqp_stages" else name
        p = os.path.join(out_dir, f"{name}.csv")
        write_csv(out[key], p)
        paths.append(p)
    return paths


def read_csv(path: str) -> List[Dict]:
    """Load a results CSV back, converting numeric/boolean fields."""
    def conv(v):
        if v in ("True", "False"):
            return v == "True"
        try:
            f = float(v)
            return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f
        except ValueError:
            return v
    with open(path, newline="") as f:
        return [{k: conv(v) for k, v in row.items()} for row in csv.DictReader(f)]
