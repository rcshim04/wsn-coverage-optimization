"""Report figures (PNG, 200 dpi). Rendering only: reads result rows/CSVs.

Encoding conventions used across all figures:
  * PSO = blue, SQP = orange (fixed, never cycled).
  * Filled marker = feasible under the ORIGINAL model; hollow 'x' = infeasible.
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from .config import Config  # noqa: E402
from .cost import sensor_costs  # noqa: E402
from .geometry import comm_edges, coverage_mask  # noqa: E402
from .problem import build_problem  # noqa: E402

COLORS = {"PSO": "#257de0", "SQP": "#f47230"}
TEXT, TEXT2, GRID = "#191919", "#666666", "#e3e3e3"
INFEAS = "#ec4647"

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": "#b5b5b5", "axes.labelcolor": TEXT, "text.color": TEXT,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
})


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _sel(rows, **kw):
    return [r for r in rows if all(np.isclose(r[k], v) if isinstance(v, float) else r[k] == v
                                   for k, v in kw.items())]


# ---------------------------------------------------------------------------
# Cost map and layouts
# ---------------------------------------------------------------------------
def _cost_grid(cfg: Config, res=201):
    x0, y0, x1, y1 = cfg.region
    gx, gy = np.meshgrid(np.linspace(x0, x1, res), np.linspace(y0, y1, res))
    k = sensor_costs(np.column_stack([gx.ravel(), gy.ravel()]), cfg).reshape(gx.shape)
    return gx, gy, k


def plot_cost_map(cfg: Config, path: str):
    problem = build_problem(cfg)
    gx, gy, k = _cost_grid(cfg)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.contourf(gx, gy, k, levels=20, cmap="Oranges")
    ax.contour(gx, gy, k, levels=10, colors="white", linewidths=0.4, alpha=0.6)
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("installation cost k(s)  [cost units]")
    for j, (mx, my, a, s) in enumerate(cfg.terrain_regions, 1):
        ax.plot(mx, my, "+", color=TEXT, ms=8, mew=1.2,
                label="terrain region centre μ_j" if j == 1 else None)
        ax.annotate(f"region {j}\na={a}, σ={s}", (mx, my), xytext=(6, 6),
                    textcoords="offset points", fontsize=7.5, color=TEXT)
    ax.plot(*cfg.base_station, marker="s", color=TEXT, ms=8, ls="", label="base station b")
    ax.annotate("base station b", cfg.base_station, xytext=(8, -12),
                textcoords="offset points", fontsize=8)
    imax = np.unravel_index(np.argmax(k), k.shape)
    ax.set_title(f"Synthetic deployment cost map  (k_max = {problem.k_max:.3f}, "
                 f"k(b) = {sensor_costs(problem.base_station[None], cfg)[0]:.3f})")
    ax.plot(gx[imax], gy[imax], marker="*", color=TEXT, ms=9, ls="",
            label=f"maximum cost k_max ({gx[imax]:.2f}, {gy[imax]:.2f})")
    ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_aspect("equal"), ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=7.5)
    return _save(fig, path)


def draw_layout(ax, cfg: Config, layout: np.ndarray, title: str, show_cost=True):
    problem = build_problem(cfg)
    S = np.asarray(layout, float)
    if show_cost:
        gx, gy, k = _cost_grid(cfg, 101)
        ax.contourf(gx, gy, k, levels=12, cmap="Oranges", alpha=0.35)
    mask = coverage_mask(S, problem.points, cfg.sensing_radius)
    P = problem.points
    ax.scatter(P[~mask, 0], P[~mask, 1], s=1.2, c="#9a9a9a", lw=0, zorder=1)
    ax.scatter(P[mask, 0], P[mask, 1], s=1.2, c="#257de0", lw=0, zorder=1)
    for s in S:
        ax.add_patch(Circle(s, cfg.sensing_radius, fill=False, ec="#257de0", lw=0.8, alpha=0.7))
    nodes = np.vstack([S, problem.base_station])
    for i, j in comm_edges(S, problem.base_station, cfg.comm_radius):
        ax.plot(*nodes[[i, j]].T, color=TEXT, lw=1.0, zorder=3)
    from .geometry import reachable_from_base
    reach = reachable_from_base(S, problem.base_station, cfg.comm_radius)
    ax.scatter(S[reach, 0], S[reach, 1], s=22, c=TEXT, zorder=4, label="sensor (connected)")
    if (~reach).any():
        ax.scatter(S[~reach, 0], S[~reach, 1], s=36, marker="x", c=INFEAS, lw=1.6,
                   zorder=4, label="sensor (disconnected)")
    ax.plot(*problem.base_station, "s", color="#ffffff", mec=TEXT, mew=1.5, ms=8, zorder=5,
            label="base station")
    x0, y0, x1, y1 = cfg.region
    ax.set_xlim(x0 - 0.3, x1 + 0.3), ax.set_ylim(y0 - 0.3, y1 + 0.3)
    ax.set_aspect("equal"), ax.grid(False)
    ax.set_title(title, fontsize=8.5)


def _layout_title(r):
    tag = "FEASIBLE" if r["feasible"] else f"INFEASIBLE ({r['violations']})"
    return (f"{r['method']}  N={r['N']}, α={r['alpha']}, seed={r['seed']}  [{tag}]\n"
            f"C={r['coverage']:.3f}  K={r['cost_norm']:.3f}  J={r['objective']:+.4f}  "
            f"K_tot={r['cost_total']:.2f}/B={r['budget']:.1f}  conn={r['n_connected']}/{r['N']}")


def plot_layout_pair(cfg: Config, runs, positions, picks: Dict[str, Dict], path: str, suptitle):
    fig, axes = plt.subplots(1, len(picks), figsize=(5.0 * len(picks), 5.4))
    for ax, (method, r) in zip(np.atleast_1d(axes), picks.items()):
        pos = _sel(positions, method=method, N=r["N"], alpha=float(r["alpha"]), seed=r["seed"])
        layout = np.array([[p["x"], p["y"]] for p in sorted(pos, key=lambda p: p["sensor"])])
        draw_layout(ax, cfg.with_overrides(n_sensors=r["N"], alpha=float(r["alpha"])),
                    layout, _layout_title(r))
    handles = [
        Line2D([], [], marker="o", ls="", color=TEXT, label="sensor (reaches b)"),
        Line2D([], [], marker="x", ls="", color=INFEAS, mew=1.6, label="sensor (disconnected)"),
        Line2D([], [], marker="s", ls="", mfc="white", mec=TEXT, label="base station"),
        Line2D([], [], color=TEXT, lw=1, label="link (d ≤ r_c)"),
        Line2D([], [], color="#257de0", lw=0.8, label="sensing disk r_s"),
        Line2D([], [], marker=".", ls="", color="#257de0", label="covered sample point"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(suptitle, fontsize=10.5)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------
def plot_convergence(runs, history, stages, n, alpha, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    seeds = sorted({r["seed"] for r in _sel(runs, N=n, alpha=alpha)})

    # (a) PSO
    ax = axes[0]
    for s in seeds:
        h = _sel(history, method="PSO", N=n, alpha=alpha, seed=s)
        it = [x["step"] for x in h]
        ax.plot(it, [x["best_fitness"] for x in h], color=COLORS["PSO"], lw=1, alpha=0.35)
        ax.plot(it, [x["best_feasible_objective"] for x in h], color=TEXT, lw=1, ls=(0, (3, 2)), alpha=0.6)
    ax.plot([], [], color=COLORS["PSO"], label="gbest penalized fitness F")
    ax.plot([], [], color=TEXT, ls=(0, (3, 2)), lw=1, label="best feasible J (original)")
    ax.set_title("(a) PSO, all seeds")
    ax.set_xlabel("iteration"), ax.set_ylabel("value"), ax.legend(fontsize=8)
    vals = [x["best_fitness"] for x in _sel(history, method="PSO", N=n, alpha=alpha) if x["step"] >= 5]
    ax.set_ylim(np.nanpercentile(vals, 1) - 0.05, np.nanmax(vals) + 0.03)

    # (b) SQP, one seed, per SLSQP iteration
    ax = axes[1]
    s0 = seeds[0]
    h = _sel(history, method="SQP", N=n, alpha=alpha, seed=s0)
    it = np.array([x["step"] for x in h])
    ax.plot(it, [-x["smooth_objective"] for x in h], color=COLORS["SQP"], ls="--", lw=1.4,
            label="smoothed objective −J~(S, β)")
    J = np.array([x["objective"] for x in h])
    feas = np.array([bool(x["feasible"]) for x in h])
    ax.plot(it, J, color=TEXT, lw=1.2, label="J(S) original")
    ax.scatter(it[~feas], J[~feas], s=8, c=INFEAS, zorder=3, label="infeasible (original)")
    st = np.array([x["stage"] for x in h])
    betas = [x["beta"] for x in _sel(stages, method="SQP", N=n, alpha=alpha, seed=s0)]
    for k in range(len(betas)):
        idx = np.flatnonzero(st == k)
        if len(idx):
            ax.axvline(it[idx[0]], color="#b5b5b5", lw=0.8)
            ax.text(it[idx[0]], 1.0, f" β={betas[k]:g}", transform=ax.get_xaxis_transform(),
                    fontsize=7, color=TEXT2, va="top")
    ax.set_title(f"(b) SQP, seed {s0}: per SLSQP iteration")
    ax.set_xlabel("cumulative SLSQP iteration"), ax.legend(fontsize=7.5, loc="lower right")

    # (c) SQP per stage, all seeds
    ax = axes[2]
    for s in seeds:
        sr = _sel(stages, method="SQP", N=n, alpha=alpha, seed=s)
        k = np.arange(len(sr))
        J = np.array([x["objective"] for x in sr])
        f = np.array([bool(x["feasible"]) for x in sr])
        ax.plot(k, J, color=COLORS["SQP"], lw=1, alpha=0.5)
        ax.scatter(k[f], J[f], s=22, color=COLORS["SQP"], zorder=3)
        ax.scatter(k[~f], J[~f], s=26, marker="x", color=INFEAS, zorder=3)
        sel = np.array([bool(x.get("selected", False)) for x in sr])
        ax.scatter(k[sel], J[sel], s=110, facecolors="none", edgecolors=TEXT, lw=1.2, zorder=4)
    ax.set_xticks(range(len(betas)), [f"{b:g}" for b in betas])
    ax.scatter([], [], color=COLORS["SQP"], label="feasible")
    ax.scatter([], [], marker="x", color=INFEAS, label="infeasible")
    ax.scatter([], [], s=110, facecolors="none", edgecolors=TEXT, lw=1.2,
               label="reported (best feasible stage)")
    ax.set_title("(c) SQP: original J at each stage end, all seeds")
    ax.set_xlabel("continuation stage β"), ax.legend(fontsize=8)
    fig.suptitle(f"Convergence (N={n}, α={alpha})", fontsize=10.5)
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------
def _strip(ax, runs, key, methods=("PSO", "SQP")):
    rng = np.random.default_rng(0)
    for i, m in enumerate(methods):
        rs = [r for r in runs if r["method"] == m]
        x = i + rng.uniform(-0.12, 0.12, len(rs))
        y = np.array([r[key] for r in rs])
        f = np.array([bool(r["feasible"]) for r in rs])
        ax.scatter(x[f], y[f], s=30, color=COLORS[m], zorder=3)
        ax.scatter(x[~f], y[~f], s=34, marker="x", color=INFEAS, lw=1.5, zorder=3)
        if f.any():
            ax.hlines(y[f].mean(), i - 0.25, i + 0.25, color=TEXT, lw=1.5)
    ax.set_xticks(range(len(methods)), methods), ax.set_xlim(-0.6, len(methods) - 0.4)


def plot_baseline_comparison(runs, n, alpha, path):
    rs = _sel(runs, N=n, alpha=alpha)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (key, lab) in zip(axes, [("coverage", "coverage C(S)"), ("cost_norm", "normalized cost K(S)"),
                                     ("objective", "objective J(S)"), ("cpu_sec", "CPU time per run [s]")]):
        _strip(ax, rs, key)
        ax.set_title(lab)
    axes[3].set_yscale("log")
    handles = [Line2D([], [], marker="o", ls="", color=COLORS["PSO"], label="PSO feasible"),
               Line2D([], [], marker="o", ls="", color=COLORS["SQP"], label="SQP feasible"),
               Line2D([], [], marker="x", ls="", color=INFEAS, mew=1.5, label="infeasible"),
               Line2D([], [], color=TEXT, lw=1.5, label="mean of feasible runs")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Baseline comparison over {len(rs) // 2} seeds (N={n}, α={alpha}); "
                 "metrics from the original model", fontsize=10.5)
    fig.tight_layout()
    return _save(fig, path)


def _by_param(summary, param, fixed):
    out = {}
    for m in ("PSO", "SQP"):
        rows = sorted(_sel(summary, method=m, **fixed), key=lambda r: r[param])
        out[m] = rows
    return out


def _num(v):
    return np.nan if v == "" else float(v)


def plot_time_vs_n(summary, alpha, path):
    d = _by_param(summary, "N", {"alpha": alpha})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for m, rows in d.items():
        x = [r["N"] for r in rows]
        axes[0].errorbar(x, [r["cpu_sec_mean"] for r in rows], yerr=[r["cpu_sec_std"] for r in rows],
                         color=COLORS[m], marker="o", ms=5, capsize=3, label=m)
        axes[1].plot(x, [r["n_obj_evals_mean"] for r in rows], color=COLORS[m], marker="o", ms=5, label=m)
    axes[0].set_yscale("log"), axes[0].set_title("CPU time per run (mean ± sd)")
    axes[0].set_ylabel("seconds"), axes[0].set_xlabel("number of sensors N (2N variables)")
    axes[1].set_yscale("log"), axes[1].set_title("objective-function evaluations per run (mean)")
    axes[1].set_xlabel("number of sensors N")
    for ax in axes:
        ax.legend()
    fig.suptitle(f"Computational cost vs problem size (α={alpha})", fontsize=10.5)
    fig.tight_layout()
    return _save(fig, path)


def plot_metrics_vs_param(summary, runs, param, fixed, path, xlabel):
    d = _by_param(summary, param, fixed)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    specs = [("objective_feas", "objective J (feasible runs)"), ("coverage_feas", "coverage C (feasible runs)"),
             ("cost_norm_feas", "normalized cost K (feasible runs)")]
    for ax, (key, title) in zip(axes, specs):
        for m, rows in d.items():
            x = np.array([r[param] for r in rows], float)
            y = np.array([_num(r[f"{key}_mean"]) for r in rows])
            e = np.array([_num(r[f"{key}_std"]) for r in rows])
            ax.errorbar(x, y, yerr=e, color=COLORS[m], marker="o", ms=5, capsize=3, label=m)
        ax.set_title(title), ax.set_xlabel(xlabel)
    ax = axes[3]
    for m, rows in d.items():
        x = [r[param] for r in rows]
        ax.plot(x, [r["feasibility_rate"] for r in rows], color=COLORS[m], marker="o", ms=5, label=m)
    ax.set_ylim(-0.05, 1.05), ax.set_title("feasibility rate (original model)"), ax.set_xlabel(xlabel)
    axes[0].legend()
    fig.suptitle("Mean ± sd over feasible runs only; points missing where no run was feasible", fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def plot_tradeoff(runs, n, path, alphas):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True, sharey=True)
    cmap = plt.get_cmap("Blues")
    shades = {a: cmap(0.35 + 0.6 * i / max(1, len(alphas) - 1)) for i, a in enumerate(alphas)}
    for ax, m in zip(axes, ("PSO", "SQP")):
        for a in alphas:
            rs = _sel(runs, method=m, N=n, alpha=float(a))
            f = [r for r in rs if r["feasible"]]
            nf = [r for r in rs if not r["feasible"]]
            ax.scatter([r["cost_norm"] for r in f], [r["coverage"] for r in f], s=30, color=shades[a],
                       ec="#4d4d4d", lw=0.4, label=f"α={a}", zorder=3)
            ax.scatter([r["cost_norm"] for r in nf], [r["coverage"] for r in nf], s=30, marker="x",
                       color=INFEAS, lw=1.2, zorder=2)
        ax.set_title(f"{m}: coverage vs cost per run (N={n})")
        ax.set_xlabel("normalized cost K(S)")
    axes[0].set_ylabel("coverage C(S)")
    axes[0].scatter([], [], marker="x", color=INFEAS, label="infeasible")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def plot_feasibility(summary, n_base, alpha_base, path):
    panels = [(p, f, lab) for p, f, lab in [("N", {"alpha": alpha_base}, "number of sensors N"),
                                            ("alpha", {"N": n_base}, "coverage weight α")]
              if len(_by_param(summary, p, f)["PSO"]) > 1]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 3.8), squeeze=False)
    axes = axes[0]
    for ax, (param, fixed, lab) in zip(axes, panels):
        d = _by_param(summary, param, fixed)
        xs = [r[param] for r in d["PSO"]]
        idx = np.arange(len(xs))
        for k, m in enumerate(("PSO", "SQP")):
            rows = d[m]
            off = (k - 0.5) * 0.38
            nr = np.array([r["n_runs"] for r in rows])
            conn = np.array([r["n_conn_violations"] for r in rows])
            bud = np.array([r["n_budget_violations"] for r in rows])
            feas = np.array([r["n_feasible"] for r in rows])
            ax.bar(idx + off, feas, 0.36, color=COLORS[m], label=f"{m} feasible")
            ax.bar(idx + off, nr - feas, 0.36, bottom=feas, color="white", ec=COLORS[m], hatch="///",
                   lw=0.8, label=f"{m} infeasible")
            for i in range(len(rows)):
                txt = []
                if conn[i]:
                    txt.append(f"c{conn[i]}")
                if bud[i]:
                    txt.append(f"b{bud[i]}")
                ax.text(idx[i] + off, nr[i] + 0.15, " ".join(txt) or "", ha="center", fontsize=7, color=TEXT2)
        ax.set_xticks(idx, [f"{x:g}" for x in xs]), ax.set_xlabel(lab)
        ax.set_ylabel("runs"), ax.set_ylim(0, max(nr) + 1.4), ax.grid(axis="x", visible=False)
        ax.set_title(f"feasible runs out of {nr[0]} ({'α' if param == 'N' else 'N'}="
                     f"{fixed.get('alpha', fixed.get('N'))})")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Constraint satisfaction (labels: c = connectivity violations, b = budget violations)",
                 fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def plot_budget_sensitivity(runs, summary, n, alpha, path):
    """Coverage, J, unused budget per run and feasibility rate vs b (B = b N)."""
    rs = _sel(runs, N=n, alpha=alpha)
    bvals = sorted({float(r["budget_per_sensor"]) for r in rs})
    idx = {b: i for i, b in enumerate(bvals)}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    rng = np.random.default_rng(0)
    for ax, (key, title) in zip(axes[:3], [("coverage", "coverage C(S)"), ("objective", "objective J(S)"),
                                           ("unused_budget", "unused budget B − K_total")]):
        for k, m in enumerate(("PSO", "SQP")):
            mr = [r for r in rs if r["method"] == m]
            x = np.array([idx[float(r["budget_per_sensor"])] for r in mr]) + (k - 0.5) * 0.3 \
                + rng.uniform(-0.07, 0.07, len(mr))
            y = np.array([r[key] for r in mr])
            f = np.array([bool(r["feasible"]) for r in mr])
            ax.scatter(x[f], y[f], s=24, color=COLORS[m], zorder=3)
            ax.scatter(x[~f], y[~f], s=28, marker="x", color=INFEAS, lw=1.4, zorder=3)
            for b in bvals:  # mean of feasible runs
                yb = [r[key] for r in mr if r["feasible"] and np.isclose(r["budget_per_sensor"], b)]
                if yb:
                    xc = idx[b] + (k - 0.5) * 0.3
                    ax.hlines(np.mean(yb), xc - 0.12, xc + 0.12, color=TEXT, lw=1.5, zorder=4)
        ax.set_title(title)
    axes[2].axhline(0, color=TEXT2, lw=1, ls=(0, (3, 2)))
    axes[2].text(len(bvals) - 0.5, 0, "budget bound", fontsize=7, color=TEXT2, ha="right", va="bottom")
    ax = axes[3]
    for m in ("PSO", "SQP"):
        srows = sorted([r for r in _sel(summary, method=m, N=n, alpha=alpha)],
                       key=lambda r: r["budget_per_sensor"])
        ax.plot([idx[float(r["budget_per_sensor"])] for r in srows], [r["feasibility_rate"] for r in srows],
                color=COLORS[m], marker="o", ms=5, label=m)
    ax.set_ylim(-0.05, 1.05), ax.set_title("feasibility rate (original model)")
    for ax in axes:
        ax.set_xticks(range(len(bvals)), [f"{b:g}N\n(B={b * n:g})" for b in bvals])
        ax.set_xlabel("budget B(N) = b·N")
    handles = [Line2D([], [], marker="o", ls="", color=COLORS["PSO"], label="PSO feasible"),
               Line2D([], [], marker="o", ls="", color=COLORS["SQP"], label="SQP feasible"),
               Line2D([], [], marker="x", ls="", color=INFEAS, mew=1.4, label="infeasible"),
               Line2D([], [], color=TEXT, lw=1.5, label="mean of feasible runs")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"Budget sensitivity (N={n}, α={alpha}); left = PSO, right = SQP at each budget",
                 fontsize=10.5)
    fig.tight_layout()
    return _save(fig, path)


def plot_sqp_selected_vs_final(cfg: Config, runs, positions_all, n, alpha, path):
    """SQP: reported (best feasible stage) vs final-stage layout for one seed where they differ."""
    cands = [r for r in _sel(runs, method="SQP", N=n, alpha=alpha)
             if r["found_feasible"] and not r["sqp_final_feasible"]]
    if not cands:
        return None
    r = cands[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4))
    c = cfg.with_overrides(n_sensors=n, alpha=alpha)
    for ax, sol, tag in [(axes[0], "reported", f"reported: best feasible stage β={r['sqp_selected_beta']:g}"),
                         (axes[1], "sqp_final_stage", f"final stage β={r['sqp_final_beta']:g}")]:
        pos = sorted(_sel(positions_all, method="SQP", N=n, alpha=alpha, seed=r["seed"], solution=sol),
                     key=lambda p: p["sensor"])
        pre = "" if sol == "reported" else "sqp_final_"
        feas = "FEASIBLE" if r[pre + "feasible"] else f"INFEASIBLE ({r[pre + 'violations']})"
        draw_layout(ax, c, np.array([[p["x"], p["y"]] for p in pos]),
                    f"SQP seed {r['seed']} — {tag}  [{feas}]\nC={r[pre + 'coverage']:.3f}  "
                    f"K={r[pre + 'cost_norm']:.3f}  J={r[pre + 'objective']:+.4f}  "
                    f"conn={r[pre + 'n_connected']}/{n}")
    fig.suptitle(f"SQP solution selection (N={n}, α={alpha}): final stage message: "
                 f"{r['sqp_final_message']}", fontsize=10)
    return _save(fig, path)


def make_all_figures(cfg: Config, results: Dict[str, List[Dict]], fig_dir: str) -> List[str]:
    runs_all, summ_all, hist, stg, pos_all = (results[k] for k in
                                              ("runs", "summary", "history", "stages", "positions"))
    n0, a0, b0 = cfg.n_sensors, cfg.alpha, cfg.budget_per_sensor
    # Main figures use the baseline budget; the budget sweep gets its own figure.
    at_b0 = lambda rows: [r for r in rows if np.isclose(r.get("budget_per_sensor", b0), b0)]  # noqa: E731
    runs, summ, hist, stg = at_b0(runs_all), at_b0(summ_all), at_b0(hist), at_b0(stg)
    pos = [p for p in at_b0(pos_all) if p.get("solution", "reported") == "reported"]
    paths = [plot_cost_map(cfg, os.path.join(fig_dir, "cost_map.png"))]
    if len({float(r["budget_per_sensor"]) for r in _sel(summ_all, N=n0, alpha=a0)}) > 1:
        paths.append(plot_budget_sensitivity(runs_all, summ_all, n0, a0,
                                             os.path.join(fig_dir, "budget_sensitivity.png")))
    p = plot_sqp_selected_vs_final(cfg, runs, at_b0(pos_all), n0, a0,
                                   os.path.join(fig_dir, "sqp_selected_vs_final.png"))
    if p:
        paths.append(p)
    base = _sel(runs, N=n0, alpha=a0)
    if base:
        picks = {}
        for m in ("PSO", "SQP"):
            rs = [r for r in base if r["method"] == m]
            feas = [r for r in rs if r["feasible"]]
            picks[m] = max(feas or rs, key=lambda r: r["objective"])
        paths.append(plot_layout_pair(cfg, runs, pos, picks, os.path.join(fig_dir, "layouts_baseline_best.png"),
                                      "Best run per method (best feasible J; best J flagged if none feasible)"))
        s0 = min(r["seed"] for r in base)
        same = {m: _sel(base, method=m, seed=s0)[0] for m in ("PSO", "SQP")}
        paths.append(plot_layout_pair(cfg, runs, pos, same, os.path.join(fig_dir, f"layouts_baseline_seed{s0}.png"),
                                      f"Final layouts from seed {s0}"))
        paths.append(plot_convergence(runs, hist, stg, n0, a0, os.path.join(fig_dir, "convergence_baseline.png")))
        paths.append(plot_baseline_comparison(runs, n0, a0, os.path.join(fig_dir, "baseline_comparison.png")))
    ns = sorted({r["N"] for r in _sel(summ, alpha=a0)})
    alphas = sorted({r["alpha"] for r in _sel(summ, N=n0)})
    if len(ns) > 1:
        paths.append(plot_time_vs_n(summ, a0, os.path.join(fig_dir, "time_vs_N.png")))
        paths.append(plot_metrics_vs_param(summ, runs, "N", {"alpha": a0}, os.path.join(fig_dir, "metrics_vs_N.png"),
                                           "number of sensors N"))
        for n in ns:
            if n == n0:
                continue
            rs = _sel(runs, N=n, alpha=a0)
            picks = {m: max([r for r in rs if r["method"] == m and r["feasible"]] or
                            [r for r in rs if r["method"] == m], key=lambda r: r["objective"]) for m in ("PSO", "SQP")}
            paths.append(plot_layout_pair(cfg, runs, pos, picks, os.path.join(fig_dir, f"layouts_N{n}_best.png"),
                                          f"Best run per method, N={n}"))
    if len(alphas) > 1:
        paths.append(plot_metrics_vs_param(summ, runs, "alpha", {"N": n0},
                                           os.path.join(fig_dir, "metrics_vs_alpha.png"), "coverage weight α"))
        paths.append(plot_tradeoff(runs, n0, os.path.join(fig_dir, "tradeoff_alpha.png"), alphas))
    if len(ns) > 1 or len(alphas) > 1:
        paths.append(plot_feasibility(summ, n0, a0, os.path.join(fig_dir, "feasibility.png")))
    return paths
