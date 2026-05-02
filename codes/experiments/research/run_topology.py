"""
run_topology.py - Network topology robustness study.

Systematically evaluates all algorithms across five communication graph
topologies:
  random   - connected random geometric graph  (stochastic, re-sampled each trial)
  ring     - cycle graph                        (deterministic)
  grid     - 2-D mesh                           (deterministic)
  complete - fully-connected averaging          (deterministic)
  star     - hub-and-spoke                      (deterministic)

For each topology x algorithm x problem the script records:
  - success rate (%) over nStart Monte-Carlo trials
  - average steps to convergence (successful trials)
  - average communication cost in MB (successful trials)
  - spectral gap of the mixing matrix W (network quality indicator)

Outputs (results_topology/):
  fig_topology/topology_bar.{pdf,png}       - grouped success-rate bar chart
  fig_topology/topology_heatmap.{pdf,png}   - steps heatmap
  data_log/topology_summary.csv             - full numerical table
  topology_report.txt                       - human-readable summary
"""

from __future__ import annotations
import os
import sys
import csv
import time
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.alg.alg_bank   import get_alg_bank
from problems.obj_factory import obj_factory
from problems.init_policy import init_policy
from utils.helper.graph   import get_topology_generators, spectral_gap
from utils.export.plot_utils import (fig_plot_topology_bar,
                                      fig_plot_topology_heatmap,
                                      fig_topology_combined_panel)


# ── Experiment settings ────────────────────────────────────────────────────
_N_AGENTS   = 10
_N_START    = 10         # Monte-Carlo trials per (topology, problem)
_MAX_IT     = 800
_TOL        = 1e-5
_TOL_TYPE   = "relF"
_MAX_TIME_S = 90.0
_MAX_COMM   = 300.0      # MB cap

# Algorithms: full MainComp comparison
_ALG_KEY = "MainComp"

# Objective problems to test
_OBJ_NAMES = ["ridge", "logsumexp", "huber", "rosenbrock", "logreg_real"]


def _run_single(alg_func, x0, prm, tol, max_time):
    """Execute one algorithm run; return (success, steps, time_s, comm_mb)."""
    try:
        t0 = time.perf_counter()
        _, out = alg_func(x0, dict(prm))
        elapsed = time.perf_counter() - t0

        if out.get("fail", False):
            return False, np.nan, elapsed, np.nan

        relF  = np.asarray(out.get("relF",  [np.nan]))
        combo = np.asarray(out.get("combo", [np.nan]))
        comm  = float(np.nanmax(out.get("commCost", [0.0])))

        converged = np.any(relF < tol) or np.any(combo < 1e-10)
        success = converged and elapsed <= max_time and comm <= _MAX_COMM

        # First iteration where convergence was achieved (not total steps run)
        if converged:
            if np.any(relF < tol):
                steps = int(np.argmax(relF < tol)) + 1
            else:
                steps = int(np.argmax(combo < 1e-10)) + 1
        else:
            steps = len(out.get("ValueF", []))
        return success, steps, elapsed, comm
    except Exception as exc:
        print(f"      ERROR: {exc}")
        return False, np.nan, np.nan, np.nan


def run_topology(obj_group: str = "core") -> None:
    """
    Run topology robustness study.

    Parameters
    ----------
    obj_group : 'core' | 'convex' | 'all' | specific objective name
    """
    results_dir = os.path.join(_root, "results_topology")
    os.makedirs(os.path.join(results_dir, "data_log"), exist_ok=True)

    # ── Objective list ────────────────────────────────────────────────────
    og = obj_group.lower()
    if og in ("core", "default"):
        # Representative subset: 2 convex (different Hessian) + 2 non-convex
        obj_names = ["ridge", "logsumexp", "huber", "rosenbrock", "logreg_real"]
    elif og == "convex":
        obj_names = ["ridge", "quadbad", "logsumexp", "huber", "linlog", "logreg_real"]
    elif og == "all":
        # 9-function main set (3x3 layout)
        obj_names = ["ridge", "quadbad", "logsumexp", "huber", "linlog", "logreg_real",
                     "rosenbrock", "styblinski_tang", "logreg_ncvr"]
    else:
        obj_names = [og]

    alg_bank        = get_alg_bank(_ALG_KEY)
    topo_generators = get_topology_generators()
    param_bank, M_alpha_policy, x0_generator = init_policy("robust")

    # Containers: {topo_name: {alg_name: success_rate}}
    topo_sr_data    = {t: {} for t in topo_generators}
    topo_steps_data = {t: {} for t in topo_generators}

    # CSV data
    csv_rows = [["Topology", "SpectralGap", "ObjName", "AlgName",
                 "SuccessRate%", "AvgSteps", "StdSteps",
                 "AvgTime(s)", "AvgComm(MB)"]]

    report_lines = [
        "=" * 70,
        "  Network Topology Robustness Study",
        f"  Agents: {_N_AGENTS}  |  Trials: {_N_START}  |  MaxIt: {_MAX_IT}",
        "=" * 70,
    ]

    for obj_name in obj_names:
        print(f"\n{'='*60}\n Objective: {obj_name}\n{'='*60}")
        report_lines += ["", f"Objective: {obj_name}", "-" * 50]

        args = param_bank.get(obj_name, [50])
        fun_list, d, L_vec, x_opt_list, f_opt_list, _, fname, fparam = \
            obj_factory(obj_name, _N_AGENTS, *args)

        policy = M_alpha_policy.get(obj_name,
                                    {"M_factor": 1.0, "alpha": 0.1, "decay": False})
        M_val   = policy["M_factor"] * float(L_vec.max())
        alp_val = policy["alpha"]    / float(L_vec.max())

        # Use per-function maxIt from policy (if larger than default _MAX_IT),
        # but cap at _MAX_IT_TOPOLOGY to prevent multi-hour runs for non-convex
        # problems where all algorithms fail anyway.
        _MAX_IT_TOPOLOGY = 1500
        maxIt_obj = min(max(_MAX_IT, policy.get("maxIt", _MAX_IT)), _MAX_IT_TOPOLOGY)
        prm_base = {
            "Nagent": _N_AGENTS, "dim": d,
            "f": fun_list, "fname": fname, "fparam": fparam,
            "M":   M_val,   "alpha":       alp_val,
            "decay_alpha": policy["decay"],
            "x_opt": x_opt_list[0] if x_opt_list else None,
            "f_opt": float(np.mean(f_opt_list)),
            "maxIt": maxIt_obj, "tol": _TOL,
            "tolType": policy.get("tolType", _TOL_TYPE),
            "verbose": False, "NC": 3, "countComm": True,
            "esom_penalty": 1.0,
            "info": 2,
        }

        x0_gen = x0_generator.get(obj_name, lambda d, _: np.random.randn(d))

        for topo_name, topo_gen in topo_generators.items():
            print(f"\n  Topology: {topo_name}")
            report_lines.append(f"\n  Topology: {topo_name}")

            # Compute spectral gap on one representative graph
            try:
                _, W_rep = topo_gen(_N_AGENTS)
                sg = spectral_gap(W_rep)
            except Exception:
                sg = np.nan
            print(f"    Spectral gap = {sg:.4f}")
            report_lines.append(f"    Spectral gap = {sg:.4f}")

            results_per_alg = {name: [] for name, _ in alg_bank}

            for s in range(_N_START):
                rng_s = np.random.RandomState(300 + s)
                np.random.set_state(rng_s.get_state())
                x0 = x0_gen(d, True)

                if topo_name == "random":
                    _, W_s = topo_gen(_N_AGENTS)   # new random graph each trial
                else:
                    _, W_s = topo_gen(_N_AGENTS)   # deterministic (same each time)

                prm = dict(prm_base)
                prm["W"] = W_s

                for alg_name, alg_func in alg_bank:
                    success, steps, elapsed, comm = _run_single(
                        alg_func, x0, prm, _TOL, _MAX_TIME_S)
                    results_per_alg[alg_name].append({
                        "success": success, "steps": steps,
                        "time": elapsed, "comm": comm,
                    })
                    status = "OK" if success else "--"
                    print(f"      [{status}] {alg_name:<16}  "
                          f"steps={steps!s:<6}  t={elapsed:.2f}s")

            # ── Aggregate statistics ─────────────────────────────────────
            hdr = (f"{'Algorithm':<18} | {'SR%':>6} | {'AvgSteps':>9} | "
                   f"{'StdSteps':>9} | {'AvgTime':>8} | {'AvgComm':>9}")
            print(f"\n{hdr}")
            print("  " + "-" * (len(hdr) - 2))
            report_lines.append(f"\n{hdr}")
            report_lines.append("-" * len(hdr))

            for alg_name, _ in alg_bank:
                runs = results_per_alg[alg_name]
                ok   = [r for r in runs if r["success"]]
                sr   = 100.0 * len(ok) / max(len(runs), 1)

                ok_steps = [r["steps"] for r in ok if np.isfinite(r["steps"])]
                ok_time  = [r["time"]  for r in ok if np.isfinite(r["time"])]
                ok_comm  = [r["comm"]  for r in ok if np.isfinite(r["comm"])]

                avg_steps = float(np.nanmean(ok_steps)) if ok_steps else np.nan
                std_steps = float(np.nanstd(ok_steps))  if len(ok_steps) > 1 else np.nan
                avg_time  = float(np.nanmean(ok_time))  if ok_time  else np.nan
                avg_comm  = float(np.nanmean(ok_comm))  if ok_comm  else np.nan

                line = (f"{alg_name:<18} | {sr:>6.1f} | {avg_steps:>9.1f} | "
                        f"{std_steps:>9.1f} | {avg_time:>8.2f} | {avg_comm:>9.2f}")
                print(f"  {line}")
                report_lines.append(line)

                # Store for plotting
                topo_sr_data[topo_name][alg_name]    = sr
                topo_steps_data[topo_name][alg_name] = avg_steps

                csv_rows.append([topo_name, round(sg, 4), obj_name, alg_name,
                                 round(sr, 1), round(avg_steps, 1),
                                 round(std_steps, 1), round(avg_time, 3),
                                 round(avg_comm, 3)])

    # ── Generate figures ──────────────────────────────────────────────────
    print("\n[Topology] Generating figures …")
    # Primary combined panel (bar + heatmap in one file - paper-ready)
    fig_topology_combined_panel(topo_sr_data, topo_steps_data, results_dir,
                                title="Algorithm robustness across network topologies")
    # Keep individual figures for supplementary material
    fig_plot_topology_bar(topo_sr_data, results_dir,
                          title="Algorithm robustness across topologies")
    fig_plot_topology_heatmap(topo_steps_data, results_dir)

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, "data_log", "topology_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(csv_rows)
    print(f"[Saved] {csv_path}")

    # ── Save TXT report ───────────────────────────────────────────────────
    report_path = os.path.join(results_dir, "topology_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines) + "\n")
    print(f"[Saved] {report_path}")
    print("[run_topology] Done.")
