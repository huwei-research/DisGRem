"""
run_parameter.py – Parameter sensitivity study.
Ported from MATLAB: run_parameter.m / run_parameter_by_alg.m

Enhancements over original
---------------------------
  Sweep #1 – M regularisation factor    (M ∈ {0.05, 0.1, 0.5, 1.0, 3.0} × L_max)
  Sweep #2 – step size α numerator      (α ∈ {0.02, 0.05, 0.1, 0.3, 0.5})
  Sweep #3 – consensus rounds NC        (NC ∈ {1, 2, 3, 5, 8})
  
  Algorithms: DisGrem, CeDisGrem, AdaDisGrem, EXTRA, DIGing
  Problems  : ridge, logsumexp, huber  (one run per (alg, sweep, config))

Outputs (results_param/)
  param_study/<alg>_M_steps_vs_<metric>.{pdf,png}    – M sweep curves
  param_study/<alg>_alpha_steps_vs_<metric>.{pdf,png} – α sweep curves
  param_study/<alg>_NC_steps_vs_<metric>.{pdf,png}    – NC sweep curves
  param_study/param_summary.csv                        – full table
"""

from __future__ import annotations
import os
import sys
import csv
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.alg.alg_bank     import get_alg_bank
from problems.obj_factory  import obj_factory
from problems.init_policy  import init_policy
from utils.helper.graph      import generate_random_graph
from utils.export.plot_utils import fig_plot_param


# ── Algorithms to sweep ───────────────────────────────────────────────────
_PARAM_ALGS = ["DisGrem", "CeDisGrem", "AdaDisGrem", "EXTRA", "DIGing"]

# ── Sweep definitions ─────────────────────────────────────────────────────
_M_VALUES  = [0.05, 0.1, 0.5, 1.0, 3.0]
_A_VALUES  = [0.02, 0.05, 0.10, 0.30, 0.50]
_NC_VALUES = [1, 2, 3, 5, 8]

# ── Test problems ─────────────────────────────────────────────────────────
_PARAM_OBJS = ["ridge", "logsumexp", "huber"]

# ── Reference parameters ─────────────────────────────────────────────────
_REF_M_FACTOR = 0.5    # baseline M for alpha/NC sweeps
_REF_ALPHA    = 0.1    # baseline alpha numerator for M/NC sweeps
_REF_NC       = 3      # baseline NC for M/alpha sweeps

_BASE_MAX_IT  = 600


def _nan_log(label: str) -> dict:
    return {
        "label": label,
        "ValueF": [np.nan], "gradNrm": [np.nan], "cons": [np.nan],
        "combo": [np.nan], "relF": [np.nan], "relX": [np.nan],
        "commCost": [np.nan], "timeCost": [np.nan],
    }


def _run_sweep(alg_func, x0, prm_base, configs, results_dir, alg_name, tag):
    """
    Execute one sweep (list of parameter configs) for a single algorithm.

    Parameters
    ----------
    configs   : list of dicts, each with keys 'label', 'M', 'alpha', 'NC'
    tag       : short string appended to filenames ('M' | 'alpha' | 'NC')
    """
    log_list = []
    for cfg in configs:
        prm = dict(prm_base)
        prm["M"]           = cfg["M"]
        prm["alpha"]       = cfg["alpha"]
        prm["NC"]          = cfg["NC"]
        prm["esom_penalty"] = cfg["M"]
        label = cfg["label"]
        print(f"    [{alg_name}] {tag}={label} …", end=" ", flush=True)
        try:
            _, out = alg_func(x0, dict(prm))
            out["label"] = label
            cv_min = float(np.nanmin(out.get("combo", [np.nan])))
            print(f"done.  min(combo)={cv_min:.2e}")
        except Exception as exc:
            print(f"FAILED: {exc}")
            out = _nan_log(label)
        log_list.append(out)

    for x_key, y_key, scale in [
        ("steps",    "combo", "loglog"),
        ("steps",    "relF",  "loglog"),
        ("commCost", "combo", "loglog"),
        ("commCost", "relF",  "loglog"),
    ]:
        fig_plot_param(log_list, f"{alg_name}_{tag}", results_dir, x_key, y_key, scale)

    return log_list


def run_parameter(alg_names: list = None) -> None:
    """
    Run the full parameter sensitivity study.

    Parameters
    ----------
    alg_names : list of algorithm names to sweep (default: _PARAM_ALGS)
    """
    if alg_names is None:
        alg_names = _PARAM_ALGS

    results_dir = os.path.join(_root, "results_param")
    os.makedirs(results_dir, exist_ok=True)

    P_base = {
        "Nagent":      10,
        "p_edge":      0.5,
        "maxIt":       _BASE_MAX_IT,
        "tol":         1e-10,
        "tolType":     "combo",
        "verbose":     False,
        "d_override":  50,
        "NC":          _REF_NC,
        "info":        2,
        "countComm":   True,
        "decay_alpha": False,
    }

    alg_registry = dict(get_alg_bank("All"))

    # CSV accumulator
    csv_rows = [["ObjName", "AlgName", "SweepType", "ParamLabel",
                 "MinCombo", "MinRelF", "Steps", "AvgComm(MB)"]]

    rng_base = np.random.RandomState(42)
    np.random.set_state(rng_base.get_state())

    for obj_name in _PARAM_OBJS:
        print(f"\n{'='*60}")
        print(f" Parameter Sensitivity – {obj_name}")
        print(f"{'='*60}")

        d = P_base["d_override"]
        fun_list, d, L_vec, x_opt_list, f_opt_list, _, fname, fparam = \
            obj_factory(obj_name, P_base["Nagent"], d, 1e-3)
        _, W = generate_random_graph(P_base["Nagent"], P_base["p_edge"])
        rng_x0 = np.random.RandomState(42)
        x0 = rng_x0.randn(d)

        L_max = float(L_vec.max())

        prm_base = dict(P_base)
        prm_base.update({
            "f": fun_list, "fname": fname, "fparam": fparam, "dim": d,
            "x_opt": x_opt_list[0] if x_opt_list else None,
            "f_opt": float(np.mean(f_opt_list)),
            "W": W, "objName": obj_name,
        })

        for alg_name in alg_names:
            if alg_name not in alg_registry:
                print(f"[Warning] '{alg_name}' not in registry – skipping.")
                continue

            alg_func = alg_registry[alg_name]
            print(f"\n  === {alg_name} on {obj_name} ===")

            # ── Sweep 1: M factor ─────────────────────────────────────────
            print(f"  [M sweep]")
            m_configs = [
                {
                    "label": f"M={m}L",
                    "M":      m * L_max,
                    "alpha":  _REF_ALPHA / L_max,
                    "NC":     _REF_NC,
                }
                for m in _M_VALUES
            ]
            m_logs = _run_sweep(alg_func, x0, prm_base,
                                m_configs, results_dir, alg_name, "M")
            for cfg, lg in zip(m_configs, m_logs):
                _record_csv(csv_rows, obj_name, alg_name, "M",
                            cfg["label"], lg)

            # ── Sweep 2: alpha ────────────────────────────────────────────
            print(f"  [α sweep]")
            a_configs = [
                {
                    "label": f"α={a}",
                    "M":      _REF_M_FACTOR * L_max,
                    "alpha":  a / L_max,
                    "NC":     _REF_NC,
                }
                for a in _A_VALUES
            ]
            a_logs = _run_sweep(alg_func, x0, prm_base,
                                a_configs, results_dir, alg_name, "alpha")
            for cfg, lg in zip(a_configs, a_logs):
                _record_csv(csv_rows, obj_name, alg_name, "alpha",
                            cfg["label"], lg)

            # ── Sweep 3: NC (consensus rounds) ────────────────────────────
            print(f"  [NC sweep]")
            nc_configs = [
                {
                    "label": f"NC={nc}",
                    "M":      _REF_M_FACTOR * L_max,
                    "alpha":  _REF_ALPHA / L_max,
                    "NC":     nc,
                }
                for nc in _NC_VALUES
            ]
            nc_logs = _run_sweep(alg_func, x0, prm_base,
                                 nc_configs, results_dir, alg_name, "NC")
            for cfg, lg in zip(nc_configs, nc_logs):
                _record_csv(csv_rows, obj_name, alg_name, "NC",
                            cfg["label"], lg)

            print(f"  → Figures saved to {results_dir}/param_study/")

    # ── Save CSV ───────────────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, "param_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(csv_rows)
    print(f"\n[Saved] {csv_path}")
    print("[run_parameter] All sweeps completed.")


def _record_csv(csv_rows, obj_name, alg_name, sweep_type, label, log):
    """Append one row to the CSV accumulator."""
    min_combo = float(np.nanmin(log.get("combo", [np.nan])))
    min_relF  = float(np.nanmin(log.get("relF",  [np.nan])))
    steps     = len(log.get("combo", []))
    avg_comm  = float(np.nanmean(log.get("commCost", [np.nan])))
    csv_rows.append([obj_name, alg_name, sweep_type, label,
                     min_combo, min_relF, steps, avg_comm])
