"""
run_ablation.py – Ablation study for DisGrem family.

Systematically isolates the contribution of each algorithmic component:
  1. NC sweep     – consensus rounds {1, 3, 5} on DisGrem
  2. Klazy sweep  – Hessian tracking period {1, 20, 80} on CeDisGrem
  3. Adaptive M   – fixed M vs adaptive LM (AdaDisGrem vs DisGrem)

Test functions: ridge, logsumexp, rosenbrock (convex + non-convex representative)
Output: results_ablation/
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

from utils.alg.alg_bank import get_alg_bank
from problems.obj_factory import obj_factory
from problems.init_policy import init_policy
from utils.helper.graph import generate_random_graph
from utils.helper.run_utils import detect_diverged

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ABLATION_OBJS = ["ridge", "logsumexp", "rosenbrock"]

_NC_VALUES    = [1, 3, 5]
_KLAZY_VALUES = [1, 20, 80]

_N_START = 3
_SEED_BASE = 500


def _run_single(alg_func, x0, prm, f0_val):
    """Run one algorithm call, return (out, elapsed)."""
    t0 = time.perf_counter()
    try:
        _, out = alg_func(x0, dict(prm))
    except Exception as e:
        out = {"fail": True, "failReason": str(e)}
    elapsed = time.perf_counter() - t0
    if detect_diverged(out, f0_val):
        out["__failed__"] = True
    return out, elapsed


def _metric(out):
    """Extract final combo and relF from a run output."""
    if out.get("__failed__") or out.get("fail"):
        return np.nan, np.nan, np.nan
    combo = np.asarray(out.get("combo", [np.nan]))
    relF = np.asarray(out.get("relF", [np.nan]))
    steps = len(out.get("ValueF", []))
    return float(np.nanmin(combo)), float(np.nanmin(relF)), steps


def _plot_sweep(results, sweep_key, sweep_vals, ylabel, obj_names, fig_dir,
                filename_base):
    """Plot a sweep: one subplot per objective."""
    from utils.export.plot_utils import _RC, _save
    n_obj = len(obj_names)
    with matplotlib.rc_context(_RC):
        fig, axes = plt.subplots(1, n_obj, figsize=(3.5 * n_obj, 3.0),
                                 squeeze=False)
        for oi, obj_name in enumerate(obj_names):
            ax = axes[0, oi]
            data = results.get(obj_name, {})
            combo_means = []
            combo_stds = []
            for sv in sweep_vals:
                runs = data.get(sv, [])
                vals = [r[0] for r in runs if np.isfinite(r[0])]
                combo_means.append(np.mean(vals) if vals else np.nan)
                combo_stds.append(np.std(vals) if len(vals) > 1 else 0.0)

            combo_means = np.array(combo_means)
            combo_stds = np.array(combo_stds)
            x = np.arange(len(sweep_vals))

            valid = np.isfinite(combo_means)
            if np.any(valid):
                ax.bar(x[valid], combo_means[valid], color=(0.0, 0.45, 0.7),
                       alpha=0.85, edgecolor="white", linewidth=0.5)
                ax.errorbar(x[valid], combo_means[valid], yerr=combo_stds[valid],
                            fmt="none", color="0.3", capsize=3, linewidth=1.0)

            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in sweep_vals], fontsize=8)
            ax.set_xlabel(sweep_key)
            ax.set_ylabel(ylabel)
            ax.set_yscale("log")
            ax.set_title(obj_name.replace("_", " ").title(), pad=4, fontsize=9)
            ax.grid(axis="y", ls="--", alpha=0.35)

        plt.tight_layout(pad=0.5)
        for ext in ("pdf", "png"):
            path = os.path.join(fig_dir, f"{filename_base}.{ext}")
            fig.savefig(path, bbox_inches="tight", dpi=300)
            print(f"[Saved] {path}")
        plt.close(fig)


def _plot_ada_comparison(results_ada, obj_names, fig_dir):
    """Bar chart comparing DisGrem (fixed M) vs AdaDisGrem (adaptive LM)."""
    from utils.export.plot_utils import _RC, _save
    n_obj = len(obj_names)
    with matplotlib.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

        for pi, (metric_idx, ylabel) in enumerate([
            (0, "min(combo)"), (2, "Steps")
        ]):
            ax = axes[pi]
            x = np.arange(n_obj)
            bar_w = 0.35

            for bi, (alg_key, label, color) in enumerate([
                ("DisGrem", "DisGrem (fixed M)", (0.0, 0.45, 0.7)),
                ("AdaDisGrem", "AdaDisGrem (adaptive LM)", (0.9, 0.25, 0.25)),
            ]):
                vals = []
                for obj_name in obj_names:
                    runs = results_ada.get(obj_name, {}).get(alg_key, [])
                    ms = [r[metric_idx] for r in runs if np.isfinite(r[metric_idx])]
                    vals.append(np.mean(ms) if ms else np.nan)
                vals = np.array(vals)
                offset = (bi - 0.5) * bar_w
                finite = np.isfinite(vals)
                heights = np.where(finite, vals, 0.0)
                ax.bar(x[finite] + offset, heights[finite], bar_w * 0.9,
                       label=label, color=color, alpha=0.85,
                       edgecolor="white", linewidth=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels([o.replace("_", " ").title() for o in obj_names],
                               rotation=25, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            if metric_idx == 0:
                ax.set_yscale("log")
            ax.grid(axis="y", ls="--", alpha=0.35)
            ax.legend(fontsize=7, frameon=True)

        fig.suptitle("Ablation: Fixed M vs Adaptive LM", fontsize=9)
        plt.tight_layout(pad=0.5)
        for ext in ("pdf", "png"):
            path = os.path.join(fig_dir, f"ablation_ada_vs_fixed.{ext}")
            fig.savefig(path, bbox_inches="tight", dpi=300)
            print(f"[Saved] {path}")
        plt.close(fig)


def run_ablation() -> None:
    """Run the full ablation study."""
    results_dir = os.path.join(_root, "results_ablation")
    fig_dir = os.path.join(results_dir, "supplement", "ablation")
    os.makedirs(fig_dir, exist_ok=True)

    param_bank, M_alpha_policy, x0_generator = init_policy("regular")
    alg_registry = dict(get_alg_bank("All"))

    P = {
        "Nagent": 10, "p_edge": 0.5, "maxIt": 500,
        "tol": 1e-12, "tolType": "combo", "verbose": False,
        "d_override": 50, "info": 2, "NC": 3,
        "countComm": True,
        "esom_penalty": 1.0,
    }

    csv_rows = [["Sweep", "ObjName", "ParamValue", "MinCombo", "MinRelF", "Steps"]]

    # ── Sweep 1: NC on DisGrem ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Ablation Sweep 1: NC (consensus rounds) on DisGrem")
    print("=" * 60)

    nc_results = {}
    for obj_name in _ABLATION_OBJS:
        nc_results[obj_name] = {}
        args = param_bank.get(obj_name, [P["d_override"]])
        fun_list, d, L_vec, x_opt_list, f_opt_list, _, fname, fparam = \
            obj_factory(obj_name, P["Nagent"], *args)
        policy = M_alpha_policy.get(obj_name, {"M_factor": 1.0, "alpha": 0.1, "decay": False})
        M_val = policy["M_factor"] * float(L_vec.max())
        alp_val = policy["alpha"] / float(L_vec.max())

        prm_base = dict(P)
        prm_base["maxIt"] = policy.get("maxIt", P["maxIt"])
        prm_base.update({
            "f": fun_list, "fname": fname, "fparam": fparam, "dim": d,
            "M": M_val, "alpha": alp_val, "decay_alpha": policy["decay"],
            "x_opt": x_opt_list[0] if x_opt_list else None,
            "f_opt": float(np.mean(f_opt_list)),
        })
        x0_gen = x0_generator.get(obj_name, lambda d, _: np.random.randn(d))

        for nc in _NC_VALUES:
            runs = []
            for s in range(_N_START):
                rng_s = np.random.RandomState(_SEED_BASE + s)
                np.random.set_state(rng_s.get_state())
                x0 = x0_gen(d, False)
                _, W = generate_random_graph(P["Nagent"], P["p_edge"])
                prm = dict(prm_base)
                prm["W"] = W
                prm["NC"] = nc
                prm["compressH"] = False
                prm["Klazy"] = 1
                f0_val = float(np.mean([fi(x0) for fi in fun_list]))
                out, elapsed = _run_single(alg_registry["DisGrem"], x0, prm, f0_val)
                m = _metric(out)
                runs.append(m)
                print(f"  [{obj_name}] NC={nc} run#{s+1}: combo={m[0]:.2e}")
                csv_rows.append(["NC", obj_name, nc, m[0], m[1], m[2]])
            nc_results[obj_name][nc] = runs

    _plot_sweep(nc_results, "NC", _NC_VALUES, "min(combo)", _ABLATION_OBJS,
                fig_dir, "ablation_NC_sweep")

    # ── Sweep 2: Klazy on CeDisGrem ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Ablation Sweep 2: Klazy (Hessian tracking period) on CeDisGrem")
    print("=" * 60)

    klazy_results = {}
    for obj_name in _ABLATION_OBJS:
        klazy_results[obj_name] = {}
        args = param_bank.get(obj_name, [P["d_override"]])
        fun_list, d, L_vec, x_opt_list, f_opt_list, _, fname, fparam = \
            obj_factory(obj_name, P["Nagent"], *args)
        policy = M_alpha_policy.get(obj_name, {"M_factor": 1.0, "alpha": 0.1, "decay": False})
        M_val = policy["M_factor"] * float(L_vec.max())
        alp_val = policy["alpha"] / float(L_vec.max())

        prm_base = dict(P)
        prm_base["maxIt"] = policy.get("maxIt", P["maxIt"])
        prm_base.update({
            "f": fun_list, "fname": fname, "fparam": fparam, "dim": d,
            "M": M_val, "alpha": alp_val, "decay_alpha": policy["decay"],
            "x_opt": x_opt_list[0] if x_opt_list else None,
            "f_opt": float(np.mean(f_opt_list)),
        })
        x0_gen = x0_generator.get(obj_name, lambda d, _: np.random.randn(d))

        for klazy in _KLAZY_VALUES:
            runs = []
            for s in range(_N_START):
                rng_s = np.random.RandomState(_SEED_BASE + s)
                np.random.set_state(rng_s.get_state())
                x0 = x0_gen(d, False)
                _, W = generate_random_graph(P["Nagent"], P["p_edge"])
                prm = dict(prm_base)
                prm["W"] = W
                prm["Klazy"] = klazy
                prm["compressH"] = True
                f0_val = float(np.mean([fi(x0) for fi in fun_list]))
                out, elapsed = _run_single(alg_registry["CeDisGrem"], x0, prm, f0_val)
                m = _metric(out)
                runs.append(m)
                print(f"  [{obj_name}] Klazy={klazy} run#{s+1}: combo={m[0]:.2e}")
                csv_rows.append(["Klazy", obj_name, klazy, m[0], m[1], m[2]])
            klazy_results[obj_name][klazy] = runs

    _plot_sweep(klazy_results, "Klazy", _KLAZY_VALUES, "min(combo)",
                _ABLATION_OBJS, fig_dir, "ablation_Klazy_sweep")

    # ── Sweep 3: Fixed M (DisGrem) vs Adaptive LM (AdaDisGrem) ─────────
    print("\n" + "=" * 60)
    print("  Ablation Sweep 3: Fixed M (DisGrem) vs Adaptive LM (AdaDisGrem)")
    print("=" * 60)

    ada_results = {}
    for obj_name in _ABLATION_OBJS:
        ada_results[obj_name] = {}
        args = param_bank.get(obj_name, [P["d_override"]])
        fun_list, d, L_vec, x_opt_list, f_opt_list, _, fname, fparam = \
            obj_factory(obj_name, P["Nagent"], *args)
        policy = M_alpha_policy.get(obj_name, {"M_factor": 1.0, "alpha": 0.1, "decay": False})
        M_val = policy["M_factor"] * float(L_vec.max())
        alp_val = policy["alpha"] / float(L_vec.max())

        prm_base = dict(P)
        prm_base["maxIt"] = policy.get("maxIt", P["maxIt"])
        prm_base.update({
            "f": fun_list, "fname": fname, "fparam": fparam, "dim": d,
            "M": M_val, "alpha": alp_val, "decay_alpha": policy["decay"],
            "x_opt": x_opt_list[0] if x_opt_list else None,
            "f_opt": float(np.mean(f_opt_list)),
        })
        x0_gen = x0_generator.get(obj_name, lambda d, _: np.random.randn(d))

        for alg_key in ("DisGrem", "AdaDisGrem"):
            runs = []
            for s in range(_N_START):
                rng_s = np.random.RandomState(_SEED_BASE + s)
                np.random.set_state(rng_s.get_state())
                x0 = x0_gen(d, False)
                _, W = generate_random_graph(P["Nagent"], P["p_edge"])
                prm = dict(prm_base)
                prm["W"] = W
                f0_val = float(np.mean([fi(x0) for fi in fun_list]))
                out, _ = _run_single(alg_registry[alg_key], x0, prm, f0_val)
                m = _metric(out)
                runs.append(m)
                print(f"  [{obj_name}] {alg_key} run#{s+1}: combo={m[0]:.2e}")
                csv_rows.append(["AdaM", obj_name, alg_key, m[0], m[1], m[2]])
            ada_results[obj_name][alg_key] = runs

    _plot_ada_comparison(ada_results, _ABLATION_OBJS, fig_dir)

    # ── Save CSV ────────────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, "ablation_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(csv_rows)
    print(f"\n[Saved] {csv_path}")
    print("[run_ablation] Done.")


if __name__ == "__main__":
    run_ablation()
