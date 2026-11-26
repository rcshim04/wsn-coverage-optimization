"""Run the PSO vs SQP experiments from the technical report (Section 3.3).

Examples
--------
    python main.py                                  # baseline only (N=10, alpha=0.7, seeds 0-7)
    python main.py --experiments all --workers 8    # baseline + N sweep + alpha sweep
    python main.py --quick                          # fast smoke run (reduced iterations)
    python main.py --experiments baseline nsweep budget --workers 8
    python main.py --figures-only --out results     # re-plot from saved CSVs

Results go to <out>/ (CSV + config.json) and <out>/figures/ (PNG).
"""

from __future__ import annotations

import os

# One BLAS thread per process: keeps per-run timings comparable when runs are
# executed in parallel worker processes. Must be set before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402

from wsn.config import DEFAULT_CONFIG  # noqa: E402
from wsn.experiment import (read_csv, run_experiments, save_results, summarize,  # noqa: E402
                            write_csv)
from wsn.problem import build_problem  # noqa: E402
from wsn.visualize import make_all_figures  # noqa: E402

ALL = ("baseline", "nsweep", "alphasweep", "budget")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="WSN coverage-cost placement: PSO vs SQP")
    p.add_argument("--experiments", nargs="+", default=["baseline"],
                   choices=[*ALL, "all"], help="which experiments to run (default: baseline)")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="override seeds (default 0..7 from config)")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel worker processes (timings are per run; default 1)")
    p.add_argument("--out", default=DEFAULT_CONFIG.output_dir)
    p.add_argument("--quick", action="store_true",
                   help="smoke run: 2 seeds, N in (5,10), alpha in (0.5,0.7), reduced iterations "
                        "(NOT the report's parameters)")
    p.add_argument("--figures-only", action="store_true", help="re-plot from existing CSVs")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    cfg = DEFAULT_CONFIG.with_overrides(output_dir=a.out)
    if a.seeds is not None:
        cfg = cfg.with_overrides(seeds=tuple(a.seeds))
    if a.quick:
        cfg = cfg.with_overrides(seeds=cfg.seeds[:2], n_values=(5, 10), alpha_values=(0.5, 0.7),
                                 budget_values=(1.4, 1.8), pso_swarm_size=20, pso_iterations=40,
                                 sqp_max_iter=60)
    experiments = ALL if "all" in a.experiments else tuple(a.experiments)
    fig_dir = os.path.join(a.out, "figures")

    if a.figures_only:
        res = {k: read_csv(os.path.join(a.out, f"{f}.csv")) for k, f in
               [("runs", "runs"), ("history", "history"),
                ("stages", "sqp_stages"), ("positions", "positions")]}
        res["summary"] = summarize(res["runs"])  # recomputed so summary.csv tracks summarize()
        write_csv(res["summary"], os.path.join(a.out, "summary.csv"))
    else:
        problem = build_problem(cfg)
        print(f"N={cfg.n_sensors} alpha={cfg.alpha} |P|={len(problem.points)} r_s={cfg.sensing_radius} "
              f"r_c={cfg.comm_radius} r_c_eff={cfg.comm_radius_eff} k_max={problem.k_max:.4f} "
              f"B(N)={cfg.budget_per_sensor}N seeds={list(cfg.seeds)} experiments={list(experiments)}")
        t0 = time.perf_counter()
        res = run_experiments(cfg, experiments, workers=a.workers)
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "config.json"), "w") as f:
            json.dump({**asdict(cfg), "k_max": problem.k_max, "experiments": list(experiments),
                       "workers": a.workers, "quick": a.quick}, f, indent=2)
        for pth in save_results(res, a.out):
            print("wrote", pth)
        print(f"experiments finished in {time.perf_counter() - t0:.1f}s")

    print_summary(res["summary"])
    for pth in make_all_figures(cfg, res, fig_dir):
        print("wrote", pth)


def print_summary(summary):
    hdr = (f"{'method':<6}{'N':>4}{'alpha':>6}{'b':>5}{'feas':>7}{'J_feas mean+-sd':>20}{'C_feas':>8}"
           f"{'K_feas':>8}{'B act/near':>11}{'cpu(s)':>9}{'obj evals':>10}{'conv':>6}  SQP selected beta")
    print(hdr), print("-" * len(hdr))
    for r in summary:
        J = (f"{r['objective_feas_mean']:+.4f}+-{r['objective_feas_std']:.4f}"
             if r["n_feasible"] else "   (none feasible)")
        C = f"{r['coverage_feas_mean']:.3f}" if r["n_feasible"] else "-"
        K = f"{r['cost_norm_feas_mean']:.3f}" if r["n_feasible"] else "-"
        print(f"{r['method']:<6}{r['N']:>4}{r['alpha']:>6g}{r['budget_per_sensor']:>5g}"
              f"{r['n_feasible']:>4}/{r['n_runs']:<2}{J:>20}{C:>8}{K:>8}"
              f"{r['n_budget_active']:>7}/{r['n_budget_near']:<3}{r['cpu_sec_mean']:>9.2f}"
              f"{r['n_obj_evals_mean']:>10.0f}{r['n_solver_success']:>4}/{r['n_runs']}"
              f"  {r.get('sqp_selected_beta_counts', '')}")
    print("feas = feasible under the original model (reported solution); B act/near = budget active / "
          "within 2%; conv = PSO: found a feasible layout, SQP: FINAL SLSQP stage reported success")

if __name__ == "__main__":
    main()
