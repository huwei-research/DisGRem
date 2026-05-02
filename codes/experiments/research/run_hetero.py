"""
run_hetero.py - Data heterogeneity experiment on logistic regression.

Compares algorithm performance under two data partitioning schemes:
  IID     - samples randomly shuffled across agents (default in obj_factory)
  Non-IID - samples sorted by label before partitioning, so each agent
            holds a skewed class distribution

Output: results_hetero/
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
from problems.init_policy import init_policy
from utils.helper.graph import generate_random_graph
from utils.helper.run_utils import detect_diverged
from utils.export.log_export import merge_logs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_N_AGENTS = 10
_N_START = 5
_MAX_IT = 1000
_TOL = 1e-8
_SEED_BASE = 800
_IOTA = 1e-3


def _log1pexp(z):
    out = np.empty_like(z)
    big = z > 30.0
    out[big] = z[big]
    out[~big] = np.log1p(np.exp(z[~big]))
    return out


def _build_logreg_funcs(A_all, b_all, partition_idx, iota):
    """Build per-agent logistic regression functions from given partition."""
    N = len(partition_idx)
    d = A_all.shape[1]
    fun_list = []
    L_vec = np.zeros(N)
    fparam = []

    for i in range(N):
        idx = partition_idx[i]
        A_i = A_all[idx]
        b_i = b_all[idx]

        def _logreg(x, A=A_i, b=b_i):
            return float(iota * 0.5 * np.dot(x, x)
                         + np.mean(_log1pexp(-b * (A @ x))))
        fun_list.append(_logreg)
        fparam.append({"A": A_i, "b": b_i, "iota": iota})
        m_i = A_i.shape[0]
        L_vec[i] = iota + 0.25 * float(np.max(np.sum(A_i ** 2, axis=1))) / m_i

    # Find consensus optimum
    from scipy.optimize import minimize
    def f_avg(x):
        return float(np.mean([fi(x) for fi in fun_list]))
    best_f, best_x = np.inf, np.zeros(d)
    for _ in range(3):
        x0 = np.random.randn(d) * 0.1
        res = minimize(f_avg, x0, method="L-BFGS-B",
                       options={"maxiter": 5000, "gtol": 1e-14})
        if res.fun < best_f:
            best_f, best_x = res.fun, res.x.copy()

    x_opts = [best_x.copy() for _ in range(N)]
    f_opts = np.array([fi(best_x) for fi in fun_list])

    return fun_list, d, L_vec, x_opts, f_opts, fparam


def _iid_partition(m_tot, N, rng):
    """Random (IID) partition of m_tot samples into N agents."""
    idx = rng.permutation(m_tot)
    blk = int(np.ceil(m_tot / N))
    return [idx[i * blk: min((i + 1) * blk, m_tot)] for i in range(N)]


def _noniid_partition(b_all, N, rng):
    """Non-IID: sort by label, then partition contiguously."""
    sorted_idx = np.argsort(b_all, kind="stable")
    # Add slight shuffle within each class to avoid identical ordering
    pos_mask = b_all[sorted_idx] > 0
    pos_idx = sorted_idx[pos_mask]
    neg_idx = sorted_idx[~pos_mask]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    sorted_idx = np.concatenate([neg_idx, pos_idx])

    m_tot = len(sorted_idx)
    blk = int(np.ceil(m_tot / N))
    return [sorted_idx[i * blk: min((i + 1) * blk, m_tot)] for i in range(N)]


def run_hetero() -> None:
    """Run IID vs Non-IID data heterogeneity comparison."""
    results_dir = os.path.join(_root, "results_hetero")
    fig_dir = os.path.join(results_dir, "supplement", "hetero")
    os.makedirs(fig_dir, exist_ok=True)

    from utils.data.load_dataset import load_dataset
    A_all, b_all = load_dataset("a9a", standardize="zscore", label_style="pm1")
    m_tot, d_raw = A_all.shape
    d_use = min(20, d_raw)
    A_all = A_all[:, :d_use]

    alg_bank = get_alg_bank("MainComp")
    groups = get_alg_groups()

    _, M_alpha_policy, _ = init_policy("regular")
    policy = M_alpha_policy.get("logreg_real",
                                {"M_factor": 1.0, "alpha": 0.1, "decay": False})

    csv_rows = [["Partition", "AlgName", "AvgSteps", "MinCombo", "MinRelF"]]
    all_logs = {}   # {partition_type: {alg_name: merged_log}}

    for part_name, part_func in [
        ("IID", lambda rng: _iid_partition(m_tot, _N_AGENTS, rng)),
        ("Non-IID", lambda rng: _noniid_partition(b_all, _N_AGENTS, rng)),
    ]:
        print(f"\n{'='*60}")
        print(f"  Data Heterogeneity: {part_name}")
        print(f"{'='*60}")

        logs_each = []
        for s in range(_N_START):
            rng = np.random.RandomState(_SEED_BASE + s)
            partition = part_func(rng)

            # Print class balance per agent
            if s == 0:
                for i, idx in enumerate(partition):
                    pos_frac = np.mean(b_all[idx] > 0)
                    print(f"  Agent {i}: {len(idx)} samples, {pos_frac*100:.0f}% positive")

            fun_list, d, L_vec, x_opts, f_opts, fparam = \
                _build_logreg_funcs(A_all, b_all, partition, _IOTA)

            M_val = policy["M_factor"] * float(L_vec.max())
            alp_val = policy["alpha"] / float(L_vec.max())

            rng_run = np.random.RandomState(_SEED_BASE + s)
            np.random.set_state(rng_run.get_state())
            _, W = generate_random_graph(_N_AGENTS, 0.5)
            x0 = np.random.randn(d) * 0.1

            prm = {
                "Nagent": _N_AGENTS, "dim": d, "p_edge": 0.5,
                "f": fun_list, "fname": "logreg_real", "fparam": fparam,
                "W": W, "M": M_val, "alpha": alp_val,
                "decay_alpha": policy["decay"],
                "x_opt": x_opts[0], "f_opt": float(np.mean(f_opts)),
                "maxIt": _MAX_IT, "tol": 1e-12, "tolType": "combo",
                "verbose": False, "NC": 3, "countComm": True,
                "esom_penalty": 1.0, "info": 2,
            }

            f0_val = float(np.mean([fi(x0) for fi in fun_list]))
            log_s = {}
            for alg_name, alg_func in alg_bank:
                t0 = time.perf_counter()
                try:
                    _, out = alg_func(x0, dict(prm))
                except Exception:
                    out = {"fail": True}
                elapsed = time.perf_counter() - t0
                if detect_diverged(out, f0_val):
                    out["__failed__"] = True
                    print(f"  [{part_name}] {alg_name} run#{s+1}: DIVERGED")
                else:
                    combo_min = float(np.nanmin(out.get("combo", [np.nan])))
                    print(f"  [{part_name}] {alg_name} run#{s+1}: "
                          f"combo={combo_min:.2e} ({elapsed:.1f}s)")
                log_s[alg_name] = out
            logs_each.append(log_s)

        log_merged = merge_logs(logs_each)
        all_logs[part_name] = log_merged

        for alg_name, _ in alg_bank:
            lg = log_merged.get(alg_name, {})
            combo = np.asarray(lg.get("combo", [np.nan]))
            relF = np.asarray(lg.get("relF", [np.nan]))
            steps = len(lg.get("ValueF", []))
            csv_rows.append([part_name, alg_name, steps,
                             f"{float(np.nanmin(combo)):.2e}",
                             f"{float(np.nanmin(relF)):.2e}"])

    # ── Plot: side-by-side convergence comparison ───────────────────────
    from utils.export.plot_utils import _RC, _save, _get_field, _cleanup_xy, _FLOOR
    with matplotlib.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=True)

        for pi, part_name in enumerate(["IID", "Non-IID"]):
            ax = axes[pi]
            ax.set_yscale("log")
            ax.grid(True, which="both", ls="--", alpha=0.35)
            ax.set_xlabel("Iteration")
            if pi == 0:
                ax.set_ylabel(r"combo = $\|\nabla F\|$ + cons")
            ax.set_title(f"LogReg ({part_name})", pad=4)

            log_merged = all_logs.get(part_name, {})
            for alg_name, _ in alg_bank:
                if alg_name not in log_merged:
                    continue
                lg = log_merged[alg_name]
                if lg.get("__failed__"):
                    continue
                y = _get_field(lg, "combo")
                if y is None:
                    continue
                x = np.arange(1, len(y) + 1, dtype=float)
                x, y = _cleanup_xy(x, y, "combo")
                if x is None:
                    continue
                label, color, ls = get_alg_style(alg_name)
                lw = 2.0 if alg_name in groups["ours"] else 1.3
                zo = 3 if alg_name in groups["ours"] else 2
                ax.plot(x, y, color=color, linestyle=ls, linewidth=lw,
                        label=label, zorder=zo, alpha=0.92)

        # Shared legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(len(handles), 5), bbox_to_anchor=(0.5, -0.02),
                   frameon=True, fontsize=7, handlelength=1.6)
        plt.tight_layout(rect=[0, 0.08, 1, 1], pad=0.5)
        _save(fig, os.path.join(fig_dir, "hetero_comparison"))
        plt.close(fig)

    # ── Bar chart: steps degradation ────────────────────────────────────
    with matplotlib.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        alg_names_plot = [an for an, _ in alg_bank
                          if an in all_logs.get("IID", {}) and an in all_logs.get("Non-IID", {})]
        x = np.arange(len(alg_names_plot))
        bar_w = 0.35

        for bi, (part_name, color, hatch) in enumerate([
            ("IID", (0.2, 0.63, 0.17), ""),
            ("Non-IID", (0.9, 0.25, 0.25), "///"),
        ]):
            vals = []
            for an in alg_names_plot:
                lg = all_logs[part_name].get(an, {})
                combo = np.asarray(lg.get("combo", [np.nan]))
                vals.append(float(np.nanmin(combo)))
            offset = (bi - 0.5) * bar_w
            ax.bar(x + offset, vals, bar_w * 0.9, label=part_name,
                   color=color, hatch=hatch, alpha=0.85,
                   edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([get_alg_style(a)[0] for a in alg_names_plot],
                           rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("min(combo)")
        ax.set_yscale("log")
        ax.set_title("Data heterogeneity impact: IID vs Non-IID", pad=4)
        ax.grid(axis="y", ls="--", alpha=0.35)
        ax.legend(frameon=True, fontsize=7)
        plt.tight_layout(pad=0.5)
        _save(fig, os.path.join(fig_dir, "hetero_bar"))
        plt.close(fig)

    # ── CSV ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, "hetero_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(csv_rows)
    print(f"\n[Saved] {csv_path}")
    print("[run_hetero] Done.")


if __name__ == "__main__":
    run_hetero()
