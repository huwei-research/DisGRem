"""
run_kappa.py – Condition number sensitivity study on quadbad.

Varies the condition number κ from 10 to 10^6 and records steps-to-convergence
for each algorithm, demonstrating the advantage of second-order methods on
increasingly ill-conditioned problems.

Outputs (results_kappa/):
  fig_kappa/kappa_sweep.{pdf,png}  – steps vs κ line plot
  kappa_summary.csv                – full numerical table
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

from utils.alg.alg_bank import get_alg_bank, get_alg_style, get_alg_groups
from problems.obj_factory import obj_factory
from problems.init_policy import init_policy
from utils.helper.graph import generate_random_graph
from utils.helper.run_utils import detect_diverged

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_KAPPA_VALUES = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
_MAX_IT = 3000
_TOL = 1e-8
_N_START = 3
_SEED_BASE = 700


def run_kappa() -> None:
    """Run condition number sweep on quadbad."""
    results_dir = os.path.join(_root, "results_kappa")
    fig_dir = os.path.join(results_dir, "supplement", "kappa")
    os.makedirs(fig_dir, exist_ok=True)

    alg_bank = get_alg_bank("MainComp")
    groups = get_alg_groups()

    P = {
        "Nagent": 10, "p_edge": 0.5, "maxIt": _MAX_IT,
        "tol": _TOL, "tolType": "relF", "verbose": False,
        "d_override": 50, "info": 2, "NC": 3,
        "countComm": True,
        "esom_penalty": 1.0,
    }

    csv_rows = [["Kappa", "AlgName", "AvgSteps", "StdSteps", "AvgTime(s)"]]

    # {alg_name: [avg_steps_per_kappa]}
    steps_data = {an: [] for an, _ in alg_bank}

    for kappa in _KAPPA_VALUES:
        print(f"\n{'='*60}")
        print(f"  κ = {kappa:.0e}")
        print(f"{'='*60}")

        # quadbad uses the second arg as conditioning parameter
        # obj_factory("quadbad", N, d, kappa_param)
        # The kappa_param is the reciprocal of the regularisation: smaller = worse conditioned
        reg_param = 1.0 / kappa

        fun_list, d, L_vec, x_opt_list, f_opt_list, _, fname, fparam = \
            obj_factory("quadbad", P["Nagent"], P["d_override"], reg_param)

        _, M_alpha_policy, x0_generator = init_policy("regular")
        policy = M_alpha_policy.get("quadbad", {"M_factor": 0.1, "alpha": 0.1, "decay": False})
        M_val = policy["M_factor"] * float(L_vec.max())
        alp_val = policy["alpha"] / float(L_vec.max())

        prm_base = dict(P)
        prm_base.update({
            "f": fun_list, "fname": fname, "fparam": fparam, "dim": d,
            "M": M_val, "alpha": alp_val, "decay_alpha": policy["decay"],
            "x_opt": x_opt_list[0] if x_opt_list else None,
            "f_opt": float(np.mean(f_opt_list)),
        })

        x0_gen = x0_generator.get("quadbad", lambda d, _: np.random.randn(d))

        for alg_name, alg_func in alg_bank:
            all_steps = []
            all_times = []
            for s in range(_N_START):
                rng_s = np.random.RandomState(_SEED_BASE + s)
                np.random.set_state(rng_s.get_state())
                x0 = x0_gen(d, False)
                _, W = generate_random_graph(P["Nagent"], P["p_edge"])
                prm = dict(prm_base)
                prm["W"] = W

                t0 = time.perf_counter()
                try:
                    _, out = alg_func(x0, dict(prm))
                except Exception:
                    out = {"fail": True}
                elapsed = time.perf_counter() - t0

                if detect_diverged(out, 0.0) or out.get("fail"):
                    all_steps.append(np.nan)
                else:
                    relF = np.asarray(out.get("relF", [np.nan]))
                    hit = np.where(relF < _TOL)[0]
                    if len(hit) > 0:
                        all_steps.append(int(hit[0]) + 1)
                    else:
                        all_steps.append(np.nan)
                all_times.append(elapsed)

            avg_s = float(np.nanmean(all_steps)) if any(np.isfinite(all_steps)) else np.nan
            std_s = float(np.nanstd(all_steps)) if sum(np.isfinite(all_steps)) > 1 else np.nan
            avg_t = float(np.nanmean(all_times))
            steps_data[alg_name].append(avg_s)

            status = f"avg_steps={avg_s:.0f}" if np.isfinite(avg_s) else "DNF"
            print(f"  {alg_name:<20} {status}")
            csv_rows.append([f"{kappa:.0e}", alg_name,
                             f"{avg_s:.1f}" if np.isfinite(avg_s) else "DNF",
                             f"{std_s:.1f}" if np.isfinite(std_s) else "-",
                             f"{avg_t:.3f}"])

    # ── Plot ────────────────────────────────────────────────────────────
    from utils.export.plot_utils import _RC, _save
    with matplotlib.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5.0, 3.5))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.35)

        for alg_name, _ in alg_bank:
            y = np.array(steps_data[alg_name], dtype=float)
            if np.all(np.isnan(y)):
                continue
            label, color, ls = get_alg_style(alg_name)
            lw = 2.1 if alg_name in groups["ours"] else 1.3
            zo = 3 if alg_name in groups["ours"] else 2
            ax.plot(_KAPPA_VALUES, y, "o-", color=color, linestyle=ls,
                    linewidth=lw, markersize=4, label=label, zorder=zo, alpha=0.92)

        ax.set_xlabel(r"Condition number $\kappa$")
        ax.set_ylabel("Steps to convergence")
        ax.set_title(r"Condition number sensitivity (QuadBad)", pad=4)
        ax.legend(loc="upper left", frameon=True, ncol=2, fontsize=7,
                  handlelength=1.6, columnspacing=0.6)
        plt.tight_layout(pad=0.5)
        _save(fig, os.path.join(fig_dir, "kappa_sweep"))
        plt.close(fig)

    # ── Save CSV ────────────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, "kappa_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(csv_rows)
    print(f"\n[Saved] {csv_path}")
    print("[run_kappa] Done.")


if __name__ == "__main__":
    run_kappa()
