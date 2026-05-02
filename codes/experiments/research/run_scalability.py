"""
run_scalability.py - Scalability study: vary N (agents) and d (dimension).

Experiments
-----------
  (A)  Fix problem, d = d_fixed, vary N ∈ {5, 10, 20, 50}
  (B)  Fix problem, N = N_fixed, vary d ∈ {10, 30, 50, 100}

For each (algorithm, size), run until convergence and record:
  - steps to convergence
  - wall-clock time (s)
  - total communication cost (MB)

Outputs (results_scalability/)
  fig_scalability/scalability_panel_<obj>.{pdf,png}   - 2 x 3 panel
  fig_scalability/scalability_N_<m>_<obj>.{pdf,png}   - individual line plots
  fig_scalability/scalability_d_<m>_<obj>.{pdf,png}
  scalability_<obj>.csv                               - full numerical table
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

from utils.alg.alg_bank     import get_alg_bank
from problems.obj_factory  import obj_factory
from problems.init_policy  import init_policy
from utils.helper.graph      import generate_random_graph
from utils.export.plot_utils import (fig_plot_scalability, fig_plot_scalability_panel,
                                      fig_scalability_multiobj_panel)


# ── Configuration ─────────────────────────────────────────────────────────
# "MainComp": Full comparison (our 4 + EXTRA + DIGing + DQM + ESOM + SONATA + NetworkGIANT)
_SCALE_ALG_KEY = "MainComp"

_N_VALS   = [3, 5, 8, 10, 15, 20, 30, 50]
_D_VALS   = [5, 10, 20, 35, 50, 75, 100, 130, 170, 200]

_D_FIXED  = 50
_N_FIXED  = 10

_MAX_IT   = 1500
_TOL      = 1e-8
_TOL_TYPE = "combo"  # overridden per-function via policy["tolType"] if available
_SEED     = 42


def _run_one(alg_func, x0: np.ndarray, prm: dict):
    """
    Run algorithm; return (steps, time_s, comm_mb).
    Returns (nan, nan, nan) on failure or non-convergence.
    """
    try:
        t0 = time.perf_counter()
        _, out = alg_func(x0, dict(prm))
        elapsed = time.perf_counter() - t0
        if out.get("fail", False):
            return np.nan, np.nan, np.nan
        combo = np.asarray(out.get("combo", [np.nan]))
        conv  = np.any(combo < _TOL)
        if not conv:
            return int(len(combo)), np.nan, np.nan
        idx   = int(np.where(combo < _TOL)[0][0])
        steps = idx + 1
        comm  = float(np.nanmax(out.get("commCost", [0.0])))
        return steps, float(elapsed), comm
    except Exception as exc:
        print(f"      ERROR: {exc}")
        return np.nan, np.nan, np.nan


def _build_prm(fun_list, d, L_vec, x_opt_list, f_opt_list, fname, fparam,
               N, W, policy):
    """Assemble the parameter dict for one (N, d) configuration."""
    M_val   = policy["M_factor"] * float(L_vec.max())
    alp_val = policy["alpha"]    / float(L_vec.max())
    tol_type = policy.get("tolType", _TOL_TYPE)
    return {
        "Nagent": N, "dim": d,
        "f": fun_list, "fname": fname, "fparam": fparam,
        "W": W, "M": M_val, "alpha": alp_val,
        "decay_alpha": policy["decay"],
        "x_opt": x_opt_list[0] if x_opt_list else None,
        "f_opt": float(np.mean(f_opt_list)),
        "maxIt": _MAX_IT, "tol": _TOL, "tolType": tol_type,
        "verbose": False, "NC": 3, "countComm": True,
        "esom_penalty": 1.0,
        "info": 2,
    }


def run_scalability(obj_name: str = "ridge") -> None:
    """
    Run scalability experiments for a given objective function.

    Parameters
    ----------
    obj_name : any single objective name recognised by obj_factory
    """
    results_dir = os.path.join(_root, "results_scalability")
    os.makedirs(results_dir, exist_ok=True)

    alg_bank = get_alg_bank(_SCALE_ALG_KEY)
    _, M_alpha_policy, x0_generator = init_policy("regular")
    policy = M_alpha_policy.get(obj_name,
                                {"M_factor": 0.1, "alpha": 0.2, "decay": False})

    csv_rows = [["Experiment", "AlgName", "SizeVal",
                 "Steps", "Time(s)", "Comm(MB)"]]

    # ── Containers ──────────────────────────────────────────────────────────
    data_N: dict = {m: {n: [] for n, _ in alg_bank}
                    for m in ["steps", "time", "comm"]}
    data_d: dict = {m: {n: [] for n, _ in alg_bank}
                    for m in ["steps", "time", "comm"]}

    # ── Experiment A: vary N ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" Scalability Study - {obj_name}  (vary N, d={_D_FIXED})")
    print(f"{'='*60}")

    for N in _N_VALS:
        print(f"\n  N = {N} …", flush=True)
        rng_N = np.random.RandomState(_SEED)
        np.random.set_state(rng_N.get_state())
        fun_list, d, L_vec, x_opt_list, f_opt_list, _, fname, fparam = \
            obj_factory(obj_name, N, _D_FIXED, 1e-3)
        _, W = generate_random_graph(N, 0.5)

        x0_gen = x0_generator.get(obj_name, lambda d, _: np.random.randn(d))
        rng_x0 = np.random.RandomState(_SEED)
        np.random.set_state(rng_x0.get_state())
        x0 = x0_gen(d, False)

        prm = _build_prm(fun_list, d, L_vec, x_opt_list, f_opt_list,
                         fname, fparam, N, W, policy)

        for alg_name, alg_func in alg_bank:
            steps, t_val, comm = _run_one(alg_func, x0, prm)
            data_N["steps"][alg_name].append(steps)
            data_N["time"][alg_name].append(t_val)
            data_N["comm"][alg_name].append(comm)
            csv_rows.append(["varyN", alg_name, N, steps, t_val, comm])

            if np.isfinite(float(steps) if steps is not None else np.nan):
                print(f"    {alg_name:<20}  steps={steps!s:<8}  t={t_val:.3g}s")
            else:
                print(f"    {alg_name:<20}  did not converge")

    # ── Experiment B: vary d ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" Scalability Study - {obj_name}  (vary d, N={_N_FIXED})")
    print(f"{'='*60}")

    for d_val in _D_VALS:
        print(f"\n  d = {d_val} …", flush=True)
        rng_d = np.random.RandomState(_SEED)
        np.random.set_state(rng_d.get_state())
        fun_list, d, L_vec, x_opt_list, f_opt_list, _, fname, fparam = \
            obj_factory(obj_name, _N_FIXED, d_val, 1e-3)
        _, W = generate_random_graph(_N_FIXED, 0.5)

        x0_gen = x0_generator.get(obj_name, lambda d, _: np.random.randn(d))
        rng_x0d = np.random.RandomState(_SEED)
        np.random.set_state(rng_x0d.get_state())
        x0 = x0_gen(d, False)

        prm = _build_prm(fun_list, d, L_vec, x_opt_list, f_opt_list,
                         fname, fparam, _N_FIXED, W, policy)

        for alg_name, alg_func in alg_bank:
            steps, t_val, comm = _run_one(alg_func, x0, prm)
            data_d["steps"][alg_name].append(steps)
            data_d["time"][alg_name].append(t_val)
            data_d["comm"][alg_name].append(comm)
            csv_rows.append(["varyd", alg_name, d_val, steps, t_val, comm])

            if np.isfinite(float(steps) if steps is not None else np.nan):
                print(f"    {alg_name:<20}  steps={steps!s:<8}  t={t_val:.3g}s")
            else:
                print(f"    {alg_name:<20}  did not converge")

    # ── Generate figures ─────────────────────────────────────────────────────
    print("\n[Scalability] Generating figures …")

    # 2 x 3 panel figure
    fig_plot_scalability_panel(data_N, data_d, _N_VALS, _D_VALS,
                                results_dir, obj_name)

    # Individual per-metric line plots (for supplementary material)
    for metric_key in ["steps", "time", "comm"]:
        pd_N = dict(data_N[metric_key]); pd_N["_x_vals"] = _N_VALS
        fig_plot_scalability(pd_N, "N", metric_key, results_dir, obj_name)

        pd_d = dict(data_d[metric_key]); pd_d["_x_vals"] = _D_VALS
        fig_plot_scalability(pd_d, "d", metric_key, results_dir, obj_name)

    # ── Save CSV ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, f"scalability_{obj_name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(csv_rows)
    print(f"\n[Saved] {csv_path}")
    print("[run_scalability] Done.")

    return {"data_N": data_N, "data_d": data_d}


def run_scalability_multi(obj_names: list = None) -> None:
    """
    Run scalability experiments for multiple objectives and produce a
    combined multi-function panel figure.

    Parameters
    ----------
    obj_names : list of objective names; defaults to ["ridge", "logsumexp"]
    """
    if obj_names is None:
        obj_names = ["ridge", "logsumexp", "logreg_real", "rosenbrock"]

    results_dir = os.path.join(_root, "results_scalability")
    all_scale_data = {}
    for obj in obj_names:
        result = run_scalability(obj)
        if result is not None:
            all_scale_data[obj] = result

    if len(all_scale_data) > 1:
        print("\n[Scalability] Generating multi-function panel …")
        fig_scalability_multiobj_panel(
            all_scale_data, _N_VALS, _D_VALS, results_dir,
            metrics=["steps", "comm"],
        )
        print(f"[Saved] {results_dir}/fig_scalability/scalability_multiobj_panel.pdf")
