"""
plot_utils.py - Publication-quality convergence plotting utilities.

Design principles (top-conference style):
  - Figures sized for IEEE/ACM double-column (3.5" wide) or full-page (7")
  - Consistent colour palette (Wong 2011 colour-blind safe + extensions)
  - Matplotlib mathtext rendering with embedded DejaVu fonts
  - Line widths and marker sizes optimised for print and screen
  - All axes: log y-scale, grid on, clean tick formatting
  - Tight layout with appropriate margins
"""

from __future__ import annotations
import os
import shutil
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from utils.alg.alg_bank import get_alg_style, get_alg_groups

# ── LaTeX availability detection ──────────────────────────────────────────
def _latex_available() -> bool:
    """Check full usetex pipeline: latex + ghostscript + responsive kpsewhich."""
    import subprocess
    if shutil.which("latex") is None:
        return False
    gs_name = "gswin64c" if os.name == "nt" else "gs"
    if shutil.which(gs_name) is None:
        return False
    try:
        subprocess.run(
            ["kpsewhich", "article.cls"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
    try:
        subprocess.run(
            [gs_name, "--version"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
    return True

_USE_USETEX = False

# ── Global rcParams for publication quality ────────────────────────────────
_RC = {
    "text.usetex":         _USE_USETEX,
    "font.family":         "serif",
    "font.serif":          ["Computer Modern Roman"] if _USE_USETEX else ["DejaVu Serif"],
    "mathtext.fontset":    "dejavuserif",
    "font.size":           9,
    "axes.titlesize":      9,
    "axes.labelsize":      9,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "legend.fontsize":     7,
    "legend.framealpha":   0.85,
    "legend.edgecolor":    "0.8",
    "lines.linewidth":     1.8,
    "lines.markersize":    4,
    "axes.linewidth":      0.8,
    "grid.linewidth":      0.5,
    "grid.alpha":          0.35,
    "savefig.dpi":         600,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.03,
    "pdf.fonttype":        42,   # embed fonts in PDF
    "ps.fonttype":         42,
}

# ── Unified style constants ─────────────────────────────────────────────────
_LW_OURS     = 2.0     # line width for proposed algorithms
_LW_BASELINE = 1.3     # line width for baselines
_GRID_ALPHA  = 0.35    # grid transparency
_LEG_FONT    = 7.5     # legend font size
_LEG_HLEN    = 1.6     # legend handle length
_LEG_COLSPC  = 0.7     # legend column spacing
_TITLE_PAD   = 4       # title padding (points)

# ── IEEE precise column widths (inches) ─────────────────────────────────────
_COL1_W = 3.487   # IEEE single-column width
_COL2_W = 7.16    # IEEE double-column width
_CELL_H = 1.8     # standard cell height for sub-panels (compact)

# Y-axis labels for known metric keys
_YLABEL = {
    "combo":    r"$\mathrm{combo}_k$",
    "relf":     r"$\mathrm{relF}$",
    "relx":     r"relX = $\|x-x^*\|/\|x_0-x^*\|$",
    "gradnrm":  r"$\|\nabla F\|$",
    "cons":     r"Consensus error",
    "valuef":   r"$F(x)$",
}

# X-axis labels
_XLABEL = {
    "steps":    "Iteration",
    "commcost": "Communication (MB)",
    "timecost": "Time (s)",
}

# Convergence floor - stop drawing once below this value
_FLOOR = 1e-13


def _running_min(arr):
    """Return a monotone non-increasing envelope of *arr*."""
    out = np.asarray(arr, dtype=float).copy()
    best = np.inf
    for i in range(len(out)):
        if np.isfinite(out[i]):
            best = min(best, out[i])
            out[i] = best
    return out


def _rc():
    return matplotlib.rc_context(_RC)


def _get_field(log: dict, key: str):
    for k, v in log.items():
        if k.lower() == key.lower():
            return np.asarray(v, dtype=float).ravel()
    return None


def _cleanup_xy(x, y, y_key):
    if x is None or y is None:
        return None, None
    K = min(len(x), len(y))
    x, y = np.array(x[:K], dtype=float), np.array(y[:K], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[valid], y[valid]
    if len(x) == 0:
        return None, None
    if "relf" in y_key.lower() or "relx" in y_key.lower():
        y = np.minimum(y, 2.0)
    # Truncate at floor
    hit = np.where(y < _FLOOR)[0]
    if len(hit):
        x = x[: hit[0] + 1]; y = y[: hit[0] + 1]
    return x, y


def _axis_labels(ax, x_key: str, y_key: str):
    xl = _XLABEL.get(x_key.lower(), x_key)
    yl = _YLABEL.get(y_key.lower(), y_key)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)


def _format_ax(ax, scale="semilogy"):
    ax.grid(True, which="both", ls="--")
    s = scale.lower()
    if "logy" in s or s == "semilogy":
        ax.set_yscale("log")
    if "logx" in s or s == "semilogx":
        ax.set_xscale("log")
    if s == "loglog":
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.yaxis.set_minor_locator(ticker.LogLocator(subs="auto"))
    ax.tick_params(which="both", direction="in", top=False, right=False)


def _add_shared_legend(fig, handles, labels, n_cols_max: int = 7) -> None:
    """Add a shared legend at the bottom of a multi-panel figure.

    Automatically computes legend height and adjusts bottom margin
    to prevent clipping.
    """
    ncols = min(len(handles), n_cols_max)
    leg = fig.legend(handles, labels, loc="lower center", ncol=ncols,
                     bbox_to_anchor=(0.5, 0.0), frameon=True,
                     handlelength=_LEG_HLEN, columnspacing=_LEG_COLSPC,
                     labelspacing=0.25, fontsize=_LEG_FONT)
    try:
        fig.canvas.draw()
        leg_h = leg.get_window_extent(
            fig.canvas.get_renderer()
        ).transformed(fig.transFigure.inverted()).height
        fig.subplots_adjust(bottom=leg_h + 0.04)
    except Exception:
        fig.subplots_adjust(bottom=0.12)


def _add_tol_line(ax, y_key: str, tol: float = 1e-12) -> None:
    """Add a horizontal dashed reference line at the convergence tolerance."""
    if y_key.lower() in ("combo", "relf"):
        ax.axhline(tol, color="0.5", ls=":", lw=0.6, zorder=1)
        ax.text(0.98, tol, r"tol", fontsize=5.5, va="bottom", ha="right",
                color="0.5", transform=ax.get_yaxis_transform())


def _add_rate_reference(ax) -> None:
    """Overlay O(1/k) and O(1/k^2) reference slope lines on log-log axes."""
    xlim = ax.get_xlim()
    if xlim[0] <= 0 or xlim[1] <= 0:
        return
    ylim = ax.get_ylim()
    if ylim[0] <= 0 or ylim[1] <= 0:
        return
    mid_x = np.sqrt(xlim[0] * xlim[1])
    ref_x = np.array([mid_x / 3, mid_x * 3])
    y_anchor = np.sqrt(ylim[0] * ylim[1]) * 0.3
    for rate, label in [(-1, r"$O(1/k)$"), (-2, r"$O(1/k^2)$")]:
        ref_y = y_anchor * (ref_x / ref_x[0]) ** rate
        ax.plot(ref_x, ref_y, ":", color="0.6", lw=0.7, zorder=1)
        ax.text(ref_x[-1], ref_y[-1], label, fontsize=5.5, color="0.6",
                va="top", ha="left")


def _save(fig, path_base: str, exts=("pdf", "png")):
    for ext in exts:
        p = f"{path_base}.{ext}"
        kw = {"dpi": 600, "bbox_inches": "tight"} if ext == "png" else {"bbox_inches": "tight"}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig.savefig(p, **kw)
            print(f"[Saved] {p}")
        except Exception as exc:
            print(f"[WARN] Failed to save {p}: {exc}")


# ── Per-algorithm markers for black-and-white print distinction ───────────
_ALG_MARKERS = {
    "DisGrem":        "o",
    "CeDisGrem":      "s",
    "AdaDisGrem":     "^",
    "CeAdaDisGrem":   "D",
    "DisGreQm":       "P",
    "EXTRA":          "v",
    "DIGing":         "<",
    "DQM":            ">",
    "ESOM":           "X",
    "SONATA":         "h",
    "NetworkGIANT":   "p",
    "DisQN":          "*",
}
_DEFAULT_MARKEVERY = 0.12  # fraction of x-range between markers


def _get_marker_props(alg_name: str, n_points: int) -> dict:
    """Return marker kwargs for convergence line plots."""
    marker = _ALG_MARKERS.get(alg_name, "o")
    every = max(1, int(n_points * _DEFAULT_MARKEVERY))
    return dict(marker=marker, markevery=every, markersize=3.5,
                markeredgewidth=0.5, markeredgecolor="white")


# ─────────────────────────────────────────────────────────────────────────────
#  Single-objective convergence figure
# ─────────────────────────────────────────────────────────────────────────────

def fig_plot_custom(log_merged: dict, alg_bank: list,
                    x_key: str, y_key: str, scale: str,
                    results_dir: str, obj_name: str = "") -> None:
    """
    Single-objective convergence figure.  Saves PDF + PNG under results_dir/supplement/single/.
    Figure width = 3.5" (single IEEE column).
    """
    fig_dir = os.path.join(results_dir, "supplement", "single")
    os.makedirs(fig_dir, exist_ok=True)

    groups = get_alg_groups()

    with _rc():
        fig, ax = plt.subplots(figsize=(_COL1_W, _CELL_H))
        _format_ax(ax, scale)

        plotted = False
        for alg_name, _ in alg_bank:
            if alg_name not in log_merged:
                continue
            L = log_merged[alg_name]
            is_failed = bool(L.get("__failed__"))

            y_raw = _get_field(L, y_key)
            if x_key.lower() == "steps":
                x_raw = (np.arange(1, len(y_raw) + 1, dtype=float)
                         if y_raw is not None else None)
            else:
                x_raw = _get_field(L, x_key)
            x, y = _cleanup_xy(x_raw, y_raw, y_key)

            label, color, ls = get_alg_style(alg_name)
            lw = _LW_OURS if alg_name in groups["ours"] else _LW_BASELINE
            zorder = 3 if alg_name in groups["ours"] else 2

            if is_failed:
                failed_label = label + r" $\times$"
                ax.plot([], [], color=(0.65, 0.65, 0.65), linestyle="none",
                        marker="x", markersize=6, markeredgewidth=1.5,
                        label=failed_label, alpha=0.85)
                plotted = True
            else:
                if x is None:
                    continue
                mk = _get_marker_props(alg_name, len(x))
                ax.plot(x, y, color=color, linestyle=ls, linewidth=lw, label=label,
                        zorder=zorder, alpha=0.92, **mk)

                y_q25 = _get_field(L, y_key + "_q25")
                y_q75 = _get_field(L, y_key + "_q75")
                if y_q25 is not None and y_q75 is not None:
                    K_band = min(len(x), len(y_q25), len(y_q75))
                    q25 = np.clip(y_q25[:K_band], _FLOOR, None)
                    q75 = np.clip(y_q75[:K_band], _FLOOR, None)
                    ax.fill_between(x[:K_band], q25, q75,
                                    color=color, alpha=0.15, zorder=zorder - 1,
                                    linewidth=0)
                plotted = True

        if not plotted:
            plt.close(fig); return

        _axis_labels(ax, x_key, y_key)
        _add_tol_line(ax, y_key)
        if scale.lower() == "loglog":
            _add_rate_reference(ax)
        ax.set_title(_clean_title(obj_name), pad=_TITLE_PAD)

        n_alg = sum(1 for a, _ in alg_bank if a in log_merged)
        ncols = 2 if n_alg > 6 else 1
        ax.legend(loc="upper right", ncol=ncols, frameon=True,
                  handlelength=_LEG_HLEN, columnspacing=_LEG_COLSPC,
                  labelspacing=0.3, fontsize=_LEG_FONT)
        plt.tight_layout(pad=0.4)

        base = os.path.join(fig_dir, f"{obj_name}_{x_key}_vs_{y_key}")
        _save(fig, base)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-objective tiled figure
# ─────────────────────────────────────────────────────────────────────────────

def fig_plot_multiobj_custom(all_logs: dict, alg_bank: list,
                             x_key: str, y_key: str, scale: str,
                             results_dir: str,
                             n_cols: int = 3) -> None:
    """
    Multi-objective tiled figure (one panel per function).

    n_cols=3 (default) produces a clean 3x3 grid for 9 standard objectives -
    suitable for direct inclusion in a paper.  Diverged algorithms are shown as
    a 'x' legend entry only (no curve drawn).
    Saves to results_dir/paper/convergence_grid/.
    """
    fig_dir = os.path.join(results_dir, "paper", "convergence_grid")
    os.makedirs(fig_dir, exist_ok=True)

    obj_names = list(all_logs.keys())
    n_obj = len(obj_names)
    if n_obj == 0:
        return

    groups = get_alg_groups()

    n_cols = max(1, min(n_cols, n_obj))
    n_rows = int(np.ceil(n_obj / n_cols))
    cell_w = _COL2_W / n_cols
    fig_w = _COL2_W
    fig_h = _CELL_H * n_rows + 0.75   # extra for shared legend

    with _rc():
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(fig_w, fig_h),
                                 squeeze=False)
        axes_flat = axes.ravel()

        handles_global, labels_global = [], []

        for jj, obj_name in enumerate(obj_names):
            ax = axes_flat[jj]
            _format_ax(ax, scale)
            ax.set_title(_clean_title(obj_name), pad=3)

            for alg_name, _ in alg_bank:
                log_obj = all_logs[obj_name]
                if alg_name not in log_obj:
                    continue
                L = log_obj[alg_name]
                is_failed = bool(L.get("__failed__"))
                label, color, ls = get_alg_style(alg_name)
                lw = _LW_OURS if alg_name in groups["ours"] else _LW_BASELINE

                if is_failed:
                    # Diverged: legend-only marker (x), no curve drawn
                    failed_label = label + r" $\times$"
                    if failed_label not in labels_global:
                        h, = ax.plot([], [], color=(0.65, 0.65, 0.65),
                                     linestyle="none", marker="x", markersize=6,
                                     markeredgewidth=1.5, label=failed_label, alpha=0.85)
                        handles_global.append(h)
                        labels_global.append(failed_label)
                    continue

                y = _get_field(L, y_key)
                if x_key.lower() == "steps":
                    x = np.arange(1, len(y) + 1, dtype=float) if y is not None else None
                else:
                    x = _get_field(L, x_key)
                x, y = _cleanup_xy(x, y, y_key)
                if x is None:
                    continue
                zo = 3 if alg_name in groups["ours"] else 2
                mk = _get_marker_props(alg_name, len(x))
                h, = ax.plot(x, y, color=color, linestyle=ls, linewidth=lw, label=label,
                             alpha=0.92, zorder=zo, **mk)

                # Shaded band (25th-75th percentile) when multiple runs are available
                y_q25 = _get_field(L, y_key + "_q25")
                y_q75 = _get_field(L, y_key + "_q75")
                if y_q25 is not None and y_q75 is not None:
                    K_band = min(len(x), len(y_q25), len(y_q75))
                    q25 = np.clip(y_q25[:K_band], _FLOOR, None)
                    q75 = np.clip(y_q75[:K_band], _FLOOR, None)
                    ax.fill_between(x[:K_band], q25, q75,
                                    color=color, alpha=0.15, zorder=zo - 1,
                                    linewidth=0)

                if label not in labels_global:
                    handles_global.append(h)
                    labels_global.append(label)

            _axis_labels(ax, x_key, y_key)
            _add_tol_line(ax, y_key)

        # Hide unused tiles
        for jj in range(n_obj, len(axes_flat)):
            axes_flat[jj].set_visible(False)

        # Shared legend at bottom; auto-wrap to at most 2 rows
        if handles_global:
            n_h = len(handles_global)
            leg_cols = min(n_h, max(5, n_cols * 2))
            legend_rows = int(np.ceil(n_h / leg_cols))
            legend_frac = 0.045 * legend_rows + 0.02
            fig.legend(handles_global, labels_global,
                       loc="lower center",
                       ncol=leg_cols,
                       bbox_to_anchor=(0.5, 0.0),
                       frameon=True,
                       handlelength=_LEG_HLEN,
                       columnspacing=_LEG_COLSPC,
                       labelspacing=0.25,
                       fontsize=_LEG_FONT)

            plt.tight_layout(rect=[0, legend_frac, 1, 1], pad=0.4)
        else:
            plt.tight_layout(pad=0.4)

        base = os.path.join(fig_dir, f"multiobj_{x_key}_vs_{y_key}")
        _save(fig, base)
        plt.close(fig)


def fig_plot_multiobj_repr(all_logs: dict, alg_bank: list,
                           x_key: str, y_key: str, scale: str,
                           results_dir: str,
                           repr_funcs: list = None) -> None:
    """
    Representative-function 2x2 panel for main text.

    Same style as fig_plot_multiobj_custom but only shows a subset of
    functions in a compact 2x2 layout.
    Saves to results_dir/paper/convergence_grid/repr_{x_key}_vs_{y_key}.
    """
    if repr_funcs is None:
        repr_funcs = ["huber", "logsumexp", "linlog", "styblinski_tang"]

    fig_dir = os.path.join(results_dir, "paper", "convergence_grid")
    os.makedirs(fig_dir, exist_ok=True)

    obj_names_avail = list(all_logs.keys())
    obj_names = [o for o in repr_funcs if o in obj_names_avail]
    n_obj = len(obj_names)
    if n_obj == 0:
        return

    groups = get_alg_groups()
    n_cols = 2
    n_rows = int(np.ceil(n_obj / n_cols))

    with _rc():
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(_COL2_W, _CELL_H * n_rows + 0.65),
                                 squeeze=False)
        axes_flat = axes.ravel()
        handles_global, labels_global = [], []

        for jj, obj_name in enumerate(obj_names):
            ax = axes_flat[jj]
            _format_ax(ax, scale)
            ax.set_title(_clean_title(obj_name), pad=3)

            for alg_name, _ in alg_bank:
                log_obj = all_logs[obj_name]
                if alg_name not in log_obj:
                    continue
                L = log_obj[alg_name]
                is_failed = bool(L.get("__failed__"))
                label, color, ls = get_alg_style(alg_name)
                lw = _LW_OURS if alg_name in groups["ours"] else _LW_BASELINE

                if is_failed:
                    failed_label = label + r" $\times$"
                    if failed_label not in labels_global:
                        h, = ax.plot([], [], color=(0.65, 0.65, 0.65),
                                     linestyle="none", marker="x", markersize=6,
                                     markeredgewidth=1.5, label=failed_label, alpha=0.85)
                        handles_global.append(h)
                        labels_global.append(failed_label)
                    continue

                y = _get_field(L, y_key)
                if x_key.lower() == "steps":
                    x = np.arange(1, len(y) + 1, dtype=float) if y is not None else None
                else:
                    x = _get_field(L, x_key)
                x, y = _cleanup_xy(x, y, y_key)
                if x is None:
                    continue
                zo = 3 if alg_name in groups["ours"] else 2
                mk = _get_marker_props(alg_name, len(x))
                h, = ax.plot(x, y, color=color, linestyle=ls, linewidth=lw, label=label,
                             alpha=0.92, zorder=zo, **mk)

                y_q25 = _get_field(L, y_key + "_q25")
                y_q75 = _get_field(L, y_key + "_q75")
                if y_q25 is not None and y_q75 is not None:
                    K_band = min(len(x), len(y_q25), len(y_q75))
                    q25 = np.clip(y_q25[:K_band], _FLOOR, None)
                    q75 = np.clip(y_q75[:K_band], _FLOOR, None)
                    ax.fill_between(x[:K_band], q25, q75,
                                    color=color, alpha=0.15, zorder=zo - 1,
                                    linewidth=0)

                if label not in labels_global:
                    handles_global.append(h)
                    labels_global.append(label)

            _axis_labels(ax, x_key, y_key)
            _add_tol_line(ax, y_key)

        for jj in range(n_obj, len(axes_flat)):
            axes_flat[jj].set_visible(False)

        if handles_global:
            n_h = len(handles_global)
            leg_cols = min(n_h, 5)
            legend_rows = int(np.ceil(n_h / leg_cols))
            legend_frac = 0.045 * legend_rows + 0.02
            fig.legend(handles_global, labels_global,
                       loc="lower center",
                       ncol=leg_cols,
                       bbox_to_anchor=(0.5, 0.0),
                       frameon=True,
                       handlelength=_LEG_HLEN,
                       columnspacing=_LEG_COLSPC,
                       labelspacing=0.25,
                       fontsize=_LEG_FONT)
            plt.tight_layout(rect=[0, legend_frac, 1, 1], pad=0.4)
        else:
            plt.tight_layout(pad=0.4)

        base = os.path.join(fig_dir, f"repr_{x_key}_vs_{y_key}")
        _save(fig, base)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Parameter sensitivity figure
# ─────────────────────────────────────────────────────────────────────────────

def fig_plot_param(log_list: list, alg_name: str,
                   results_dir: str, x_key: str,
                   y_key: str, scale: str = "loglog") -> None:
    """Parameter sensitivity figure for one algorithm."""
    fig_dir = os.path.join(results_dir, "supplement", "param_study")
    os.makedirs(fig_dir, exist_ok=True)

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.1, 0.9, max(len(log_list), 1)))

    with _rc():
        fig, ax = plt.subplots(figsize=(_COL1_W, _CELL_H))
        _format_ax(ax, scale)

        for idx, L in enumerate(log_list):
            if L is None:
                continue
            y = _get_field(L, y_key)
            if x_key.lower() == "steps":
                x = np.arange(1, len(y) + 1, dtype=float) if y is not None else None
            else:
                x = _get_field(L, x_key)
            x, y = _cleanup_xy(x, y, y_key)
            if x is None:
                continue
            label = L.get("label", f"run {idx + 1}")
            ax.plot(x, y, color=colors[idx], linestyle="-", linewidth=1.8, label=label)

        _axis_labels(ax, x_key, y_key)
        ax.set_title(f"{alg_name}", pad=_TITLE_PAD)
        ax.legend(loc="upper right", frameon=True, handlelength=_LEG_HLEN,
                  fontsize=_LEG_FONT)
        plt.tight_layout(pad=0.4)

        base = os.path.join(fig_dir, f"{alg_name}_{x_key}_vs_{y_key}")
        _save(fig, base)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Robust test: bar chart of success rates + box plot of steps
# ─────────────────────────────────────────────────────────────────────────────

def fig_plot_robust_summary(robust_data: dict, obj_name: str,
                             results_dir: str) -> None:
    """
    Dual-panel bar chart:
      Left  - Success rate (%) per algorithm
      Right - Box plot of steps-to-convergence (successful runs only)

    Parameters
    ----------
    robust_data : dict  alg_name → {"sr": float, "steps_ok": list, "comm_ok": list}
    obj_name    : str   objective function name (for title/filename)
    results_dir : str
    """
    fig_dir = os.path.join(results_dir, "supplement", "robust_detail")
    os.makedirs(fig_dir, exist_ok=True)

    alg_names = list(robust_data.keys())
    n = len(alg_names)
    if n == 0:
        return

    sr_vals    = [robust_data[a]["sr"]       for a in alg_names]
    steps_data = [robust_data[a]["steps_ok"] for a in alg_names]
    colors     = [get_alg_style(a)[1]        for a in alg_names]
    labels     = [get_alg_style(a)[0]        for a in alg_names]

    with _rc():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(_COL2_W, _CELL_H))

        # Left: success rate bar chart
        xpos = np.arange(n)
        bars = ax1.bar(xpos, sr_vals, color=colors, edgecolor="white",
                       linewidth=0.6, zorder=3)
        ax1.set_xticks(xpos)
        ax1.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        ax1.set_ylim(0, 105)
        ax1.set_ylabel("Success rate (%)")
        ax1.set_title(_clean_title(obj_name) + " - robustness", pad=_TITLE_PAD)
        ax1.grid(axis="y", ls="--", alpha=0.4)
        ax1.axhline(100, color="0.6", ls=":", lw=0.8)
        for bar, val in zip(bars, sr_vals):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, val + 1,
                         f"{val:.0f}", ha="center", va="bottom", fontsize=6)

        # Right: box plot of steps
        valid = [(s, c) for s, c in zip(steps_data, colors) if len(s) > 0]
        if valid:
            bp = ax2.boxplot([v[0] for v in valid],
                             patch_artist=True,
                             medianprops=dict(color="white", linewidth=1.5),
                             whiskerprops=dict(linewidth=0.8),
                             capprops=dict(linewidth=0.8),
                             flierprops=dict(marker="x", markersize=4))
            for patch, (_, c) in zip(bp["boxes"], valid):
                patch.set_facecolor(c)
                patch.set_alpha(0.75)
            ax2.set_xticks(range(1, len(valid) + 1))
            ax2.set_xticklabels([get_alg_style(a)[0]
                                  for a in alg_names if len(robust_data[a]["steps_ok"]) > 0],
                                 rotation=40, ha="right", fontsize=7)
        ax2.set_ylabel("Steps to convergence")
        ax2.grid(axis="y", ls="--", alpha=0.4)

        plt.tight_layout(pad=0.5)
        base = os.path.join(fig_dir, f"robust_{obj_name}")
        _save(fig, base)
        plt.close(fig)


def fig_plot_robust_heatmap(all_robust: dict, results_dir: str) -> None:
    """
    Heatmap: rows = algorithms, cols = objective functions, values = success rate (%).
    """
    fig_dir = os.path.join(results_dir, "supplement", "heatmaps")
    os.makedirs(fig_dir, exist_ok=True)

    if not all_robust:
        return

    obj_names = list(all_robust.keys())
    # Collect all algorithm names from first objective
    alg_names = list(next(iter(all_robust.values())).keys())

    data = np.full((len(alg_names), len(obj_names)), np.nan)
    for jj, obj in enumerate(obj_names):
        for ii, alg in enumerate(alg_names):
            if alg in all_robust[obj]:
                data[ii, jj] = all_robust[obj][alg]["sr"]

    labels_alg = [get_alg_style(a)[0] for a in alg_names]
    labels_obj = [_clean_title(o, short=True) for o in obj_names]

    with _rc():
        fig_h = max(2.5, 0.35 * len(alg_names) + 0.8)
        fig_w = max(5.0, 0.55 * len(obj_names) + 1.2)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(data, aspect="auto", cmap="YlGnBu",
                       vmin=0, vmax=100, interpolation="nearest")
        ax.set_xticks(range(len(obj_names)))
        ax.set_xticklabels(labels_obj, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(alg_names)))
        ax.set_yticklabels(labels_alg, fontsize=8)

        # Annotate cells
        for i in range(len(alg_names)):
            for j in range(len(obj_names)):
                v = data[i, j]
                if np.isfinite(v):
                    txt = f"{v:.0f}"
                    color = "white" if v < 30 or v > 80 else "black"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=6.5, color=color, fontweight="bold")

        plt.colorbar(im, ax=ax, label="Success rate (%)", shrink=0.8)
        ax.set_title("Robustness heatmap (success rate %)", pad=5)
        plt.tight_layout(pad=0.4)

        base = os.path.join(fig_dir, "robust_heatmap")
        _save(fig, base)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-scenario x multi-function robust panel
# ─────────────────────────────────────────────────────────────────────────────

# Per-objective colour palettes (each function gets its own hue family)
_OBJ_PALETTES = {
    "ridge":           "Blues",
    "quadbad":         "Purples",
    "logsumexp":       "Oranges",
    "huber":           "Greens",
    "linlog":          "Reds",
    "logreg_real":     "YlOrBr",
    "rosenbrock":      "PuBu",
    "styblinski_tang": "BuPu",
    "logreg_ncvr":     "YlGn",
}
_PALETTE_DEFAULT = "Greys"


def fig_robust_multiobj_panel(all_robust_by_scenario: dict,
                               alg_bank: list,
                               scenario_order: list,
                               obj_order: list,
                               results_dir: str) -> None:
    """
    Large panel figure: rows = scenarios, columns = objective functions.
    Each cell contains a horizontal bar chart of success rates (%) for all
    algorithms.  Each column uses its own colour palette so functions are
    visually distinguishable at a glance.

    Parameters
    ----------
    all_robust_by_scenario : {scenario: {obj: {alg: {"sr": float, ...}}}}
    alg_bank               : list of (name, callable) in display order
    scenario_order         : list of scenario names (row order)
    obj_order              : list of objective names (column order)
    results_dir            : output root directory
    """
    import matplotlib.cm as mcm

    fig_dir = os.path.join(results_dir, "paper", "robust_panel")
    os.makedirs(fig_dir, exist_ok=True)

    n_sc  = len(scenario_order)
    n_obj = len(obj_order)
    if n_sc == 0 or n_obj == 0:
        return

    alg_names  = [an for an, _ in alg_bank]
    n_alg      = len(alg_names)
    groups     = get_alg_groups()

    # Build colour maps: for each objective, n_alg shades from its palette
    def _palette_colors(obj_name, n):
        cmap_name = _OBJ_PALETTES.get(obj_name, _PALETTE_DEFAULT)
        cmap = mcm.get_cmap(cmap_name)
        # Use range [0.30, 0.85] to avoid too-light or too-dark extremes
        return [cmap(0.30 + 0.55 * i / max(n - 1, 1)) for i in range(n)]

    cell_h = max(1.4, 0.22 * n_alg + 0.4)
    cell_w = 2.4
    fig_h  = cell_h * n_sc + 0.5
    fig_w  = cell_w * n_obj + 0.8

    with _rc():
        fig, axes = plt.subplots(n_sc, n_obj,
                                 figsize=(fig_w, fig_h),
                                 squeeze=False)

        for ri, sc_name in enumerate(scenario_order):
            sc_data = all_robust_by_scenario.get(sc_name, {})
            for ci, obj_name in enumerate(obj_order):
                ax = axes[ri, ci]
                obj_data = sc_data.get(obj_name, {})
                colors   = _palette_colors(obj_name, n_alg)

                y_pos = np.arange(n_alg)
                sr_vals = []
                for ai, an in enumerate(alg_names):
                    sr = float(obj_data.get(an, {}).get("sr", 0.0))
                    sr_vals.append(sr)

                # Draw bars (reversed so first alg is at top)
                for ai in range(n_alg):
                    ax.barh(y_pos[ai], sr_vals[ai],
                            color=colors[ai],
                            edgecolor="white", linewidth=0.4,
                            alpha=0.88, zorder=2,
                            height=0.72)

                ax.set_xlim(0, 105)
                ax.set_yticks(y_pos)
                if ci == 0:
                    ax.set_yticklabels(
                        [get_alg_style(an)[0] for an in alg_names],
                        fontsize=5.5)
                else:
                    ax.set_yticklabels([])
                ax.tick_params(axis="x", labelsize=5.5)
                ax.set_xlabel("Success %", fontsize=5.5) if ri == n_sc - 1 else None
                ax.grid(True, axis="x", ls="--", alpha=0.30)

                # Row label (left-most column only)
                if ci == 0:
                    ax.set_ylabel(sc_name, fontsize=6.5, labelpad=3)

                # Column title (top row only)
                if ri == 0:
                    ax.set_title(_clean_title(obj_name, short=True),
                                 pad=3, fontsize=7)

        plt.tight_layout(pad=0.4, h_pad=0.35, w_pad=0.25)
        base = os.path.join(fig_dir, "robust_panel")
        _save(fig, base)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Scalability figures
# ─────────────────────────────────────────────────────────────────────────────

def fig_plot_scalability(scale_data: dict, vary_key: str,
                          metric_key: str, results_dir: str,
                          obj_name: str = "ridge") -> None:
    """
    Line plot: x = N or d (vary_key values), y = steps/time/comm (metric_key).

    Parameters
    ----------
    scale_data : dict  alg_name → list of metric values (one per vary_key value)
    vary_key   : 'N' or 'd'
    metric_key : 'steps' | 'time' | 'comm'
    vary_vals  : list of x-axis values (passed via scale_data key "_x_vals")
    """
    fig_dir = os.path.join(results_dir, "supplement", "scalability_lines")
    os.makedirs(fig_dir, exist_ok=True)

    x_vals = scale_data.pop("_x_vals", None)
    if x_vals is None:
        return

    groups = get_alg_groups()
    _ylabels = {"steps": "Steps to convergence",
                "time":  "Time (s)",
                "comm":  "Communication (MB)"}

    with _rc():
        fig, ax = plt.subplots(figsize=(_COL1_W, _CELL_H))
        ax.grid(True, which="both", ls="--", alpha=_GRID_ALPHA)
        ax.set_yscale("log")

        for alg_name, vals in scale_data.items():
            if not vals or all(v is None or np.isnan(v) for v in vals):
                continue
            y = np.array([v if v is not None else np.nan for v in vals], dtype=float)
            label, color, ls = get_alg_style(alg_name)
            lw = _LW_OURS if alg_name in groups["ours"] else _LW_BASELINE
            ax.plot(x_vals, y, "o", color=color, linestyle=ls, linewidth=lw,
                    label=label, markersize=4,
                    zorder=3 if alg_name in groups["ours"] else 2)

        ax.set_xlabel(f"Number of {'agents' if vary_key == 'N' else 'variables'} "
                      f"({'$N$' if vary_key == 'N' else '$d$'})")
        ax.set_ylabel(_ylabels.get(metric_key, metric_key))
        ax.set_title(f"Scalability in {vary_key} ({_clean_title(obj_name)})", pad=_TITLE_PAD)
        ax.set_xticks(x_vals)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.legend(loc="upper left", frameon=True, ncol=2,
                  handlelength=_LEG_HLEN, columnspacing=_LEG_COLSPC,
                  fontsize=_LEG_FONT)
        plt.tight_layout(pad=0.4)

        base = os.path.join(fig_dir, f"scalability_{vary_key}_{metric_key}_{obj_name}")
        _save(fig, base)
        plt.close(fig)
        scale_data["_x_vals"] = x_vals   # restore


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

_TITLE_MAP = {
    "ridge":           "Ridge",
    "quadbad":         "QuadBad",
    "logsumexp":       "LogSumExp",
    "huber":           "Huber",
    "linlog":          "LinLog",
    "rosenbrock":      "Rosenbrock",
    "styblinski_tang": "Styblinski-Tang",
    "logreg_real":     "LogReg (real)",
    "logreg_ncvr":     "LogReg-NCVR",
}

_SHORT_MAP = {
    "ridge":           "Ridge",
    "quadbad":         "QuadBad",
    "logsumexp":       "LSE",
    "huber":           "Huber",
    "linlog":          "LinLog",
    "rosenbrock":      "Rosen",
    "styblinski_tang": "ST",
    "logreg_real":     "LR-real",
    "logreg_ncvr":     "LR-ncvr",
}


def _clean_title(name: str, short: bool = False) -> str:
    m = _SHORT_MAP if short else _TITLE_MAP
    return m.get(name.lower(), name)


# ─────────────────────────────────────────────────────────────────────────────
#  Performance Profile (Dolan & Moré, 2002)
# ─────────────────────────────────────────────────────────────────────────────

def fig_plot_performance_profile(perf_matrix: dict, metric_label: str,
                                  results_dir: str,
                                  title_suffix: str = "") -> None:
    """
    Dolan-Moré performance profile.

    For each algorithm a and problem p let r_{a,p} = t_{a,p} / min_a t_{a,p}.
    The performance profile is  rho_a(tau) = (1/|P|) |{p : r_{a,p} <= tau}|.

    Parameters
    ----------
    perf_matrix  : {alg_name: {obj_name: float}}
                   Metric value for each (algorithm, problem) pair.
                   Use np.nan to indicate failure / non-convergence.
    metric_label : short string describing the metric (for axis labels)
    results_dir  : output directory
    title_suffix : appended to figure title (e.g. objective name)
    """
    fig_dir = os.path.join(results_dir, "supplement", "perf_profile_single")
    os.makedirs(fig_dir, exist_ok=True)

    alg_names = list(perf_matrix.keys())
    if not alg_names:
        return
    obj_names = list(perf_matrix[alg_names[0]].keys())
    n_p = len(obj_names)
    if n_p == 0:
        return

    # Build performance matrix  T[ai, pi]
    T = np.full((len(alg_names), n_p), np.nan)
    for ai, alg in enumerate(alg_names):
        for pi, obj in enumerate(obj_names):
            T[ai, pi] = perf_matrix[alg].get(obj, np.nan)

    # Performance ratio  R[ai, pi] = T[ai, pi] / min_a T[ai, pi]
    with np.errstate(all="ignore"):
        t_best = np.nanmin(T, axis=0)              # shape (n_p,)
        t_best[t_best <= 0] = np.nan
        R = T / t_best[np.newaxis, :]

    finite_R = R[np.isfinite(R)]
    if len(finite_R) == 0:
        return
    tau_max = min(float(np.nanmax(finite_R)) * 1.1, 50.0)
    tau = np.linspace(1.0, tau_max, 600)

    groups = get_alg_groups()

    with _rc():
        fig, ax = plt.subplots(figsize=(_COL1_W, _CELL_H))
        ax.set_xlim(1.0, tau_max)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, which="both", ls="--", alpha=_GRID_ALPHA)

        for ai, alg_name in enumerate(alg_names):
            r_a = R[ai, :]
            valid = r_a[np.isfinite(r_a)]
            rho = np.array([float(np.sum(valid <= t)) / float(n_p)
                            for t in tau])
            label, color, ls = get_alg_style(alg_name)
            lw = _LW_OURS if alg_name in groups["ours"] else _LW_BASELINE
            zorder = 3 if alg_name in groups["ours"] else 2
            ax.step(tau, rho, color=color, linestyle=ls, linewidth=lw,
                    label=label, where="post", zorder=zorder, alpha=0.92)

        ax.set_xlabel("Performance ratio")
        ax.set_ylabel("Fraction solved")
        title = f"Performance profile ({metric_label})" if metric_label else "Performance profile"
        if title_suffix:
            title += f"\n{title_suffix}"
        ax.set_title(title, pad=_TITLE_PAD)

        n_alg = len(alg_names)
        ncols = 2 if n_alg > 6 else 1
        ax.legend(loc="lower right", ncol=ncols, frameon=True,
                  handlelength=_LEG_HLEN, columnspacing=_LEG_COLSPC,
                  labelspacing=0.3, fontsize=_LEG_FONT)
        plt.tight_layout(pad=0.4)

        suf = metric_label.lower().replace(" ", "_") if metric_label else "perf"
        base = os.path.join(fig_dir, f"performance_profile_{suf}")
        _save(fig, base)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Tolerance-based Performance / Data Profile Panel
# ─────────────────────────────────────────────────────────────────────────────

def _infer_dim(obj_log: dict, fallback: int = 30) -> int:
    """Infer problem dimension from xBar shape in any algorithm's merged log."""
    for v in obj_log.values():
        if not isinstance(v, dict):
            continue
        xb = v.get("xBar")
        if xb is not None:
            xb = np.asarray(xb)
            if xb.ndim == 2 and xb.shape[1] > 0:
                return int(xb.shape[1])
    return fallback


def fig_perf_profiles_tol_panel(all_logs: dict, alg_bank: list,
                                 results_dir: str,
                                 tol_levels: list = None,
                                 all_individual: dict = None) -> None:
    """
    6-panel figure (2 rows x 3 columns):
      Row 0 -- Performance profiles (Dolan-More)
      Row 1 -- Data profiles (More-Wild, simplex-gradient normalised)

    When *all_individual* is supplied (``{obj: [run0, run1, ...]}``, each
    run a ``{alg: log_dict}``), every (objective, MC-run) pair becomes a
    separate problem instance, giving much smoother step curves.
    """
    if tol_levels is None:
        tol_levels = [1e-3, 1e-6, 1e-9]

    fig_dir = os.path.join(results_dir, "paper", "perf_profiles")
    os.makedirs(fig_dir, exist_ok=True)

    alg_names = [an for an, _ in alg_bank]
    obj_names = list(all_logs.keys())
    if not obj_names or not alg_names:
        return

    groups = get_alg_groups()

    use_individual = (all_individual is not None
                      and len(all_individual) > 0)

    if use_individual:
        instance_ids = []
        instance_dims = []
        for on in obj_names:
            dim = _infer_dim(all_logs[on])
            runs = all_individual.get(on, [])
            for ri in range(len(runs)):
                instance_ids.append((on, ri))
                instance_dims.append(dim)
        n_p = len(instance_ids)
    else:
        n_p = len(obj_names)
        instance_dims = [_infer_dim(all_logs[on]) for on in obj_names]

    if n_p == 0:
        return

    def _first_hit_steps(log, tol):
        relF_arr = np.asarray(log.get("relF", []), dtype=float)
        if len(relF_arr) == 0:
            return np.nan
        hit = np.where(relF_arr < tol)[0]
        return float(hit[0] + 1) if len(hit) else np.nan

    def _build_matrix(tol):
        T = np.full((len(alg_names), n_p), np.nan)
        if use_individual:
            for ai, an in enumerate(alg_names):
                for pi, (on, ri) in enumerate(instance_ids):
                    lg = all_individual[on][ri].get(an, {})
                    if lg.get("__failed__"):
                        continue
                    T[ai, pi] = _first_hit_steps(lg, tol)
        else:
            for ai, an in enumerate(alg_names):
                for pi, on in enumerate(obj_names):
                    lg = all_logs[on].get(an, {})
                    if lg.get("__failed__"):
                        continue
                    T[ai, pi] = _first_hit_steps(lg, tol)
        return T

    def _build_matrix_mw(tol):
        T_raw = _build_matrix(tol)
        for pi in range(n_p):
            T_raw[:, pi] /= (instance_dims[pi] + 1)
        return T_raw

    def _plot_perf_profile(ax, T, title_str):
        n_inst = T.shape[1]
        with np.errstate(all="ignore"):
            t_best = np.nanmin(T, axis=0)
            t_best[t_best <= 0] = np.nan
            R = T / t_best[np.newaxis, :]

        finite_R = R[np.isfinite(R)]
        if len(finite_R) == 0:
            ax.set_visible(False)
            return
        tau_max = min(float(np.nanmax(finite_R)) * 1.05, 50.0)
        tau = np.linspace(1.0, tau_max, 500)

        ax.set_xlim(1.0, tau_max)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, which="both", ls="--", alpha=_GRID_ALPHA)
        ax.set_xlabel("Performance ratio", fontsize=7.5)
        ax.set_ylabel("Fraction solved", fontsize=7.5)
        ax.set_title(title_str, pad=_TITLE_PAD, fontsize=8)

        for ai, an in enumerate(alg_names):
            r_a = R[ai, :]
            valid = r_a[np.isfinite(r_a)]
            rho = np.array([float(np.sum(valid <= t)) / float(n_inst)
                            for t in tau])
            label, color, ls = get_alg_style(an)
            lw = _LW_OURS if an in groups["ours"] else _LW_BASELINE
            zo = 3 if an in groups["ours"] else 2
            ax.step(tau, rho, color=color, linestyle=ls, linewidth=lw,
                    label=label, where="post", zorder=zo, alpha=0.92)

    def _plot_data_profile(ax, T, title_str):
        n_inst = T.shape[1]
        finite = T[np.isfinite(T)]
        if len(finite) == 0:
            ax.set_visible(False)
            return
        kappa_max = float(np.nanmax(finite)) * 1.05
        kappa = np.linspace(0, kappa_max, 500)

        ax.set_xlim(0, kappa_max)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, which="both", ls="--", alpha=_GRID_ALPHA)
        ax.set_xlabel("Normalized budget", fontsize=7.5)
        ax.set_ylabel("Fraction solved", fontsize=7.5)
        ax.set_title(title_str, pad=_TITLE_PAD, fontsize=8)

        for ai, an in enumerate(alg_names):
            t_a = T[ai, :]
            rho = np.array([float(np.sum(t_a[np.isfinite(t_a)] <= k))
                            / float(n_inst)
                            for k in kappa])
            label, color, ls = get_alg_style(an)
            lw = _LW_OURS if an in groups["ours"] else _LW_BASELINE
            zo = 3 if an in groups["ours"] else 2
            ax.step(kappa, rho, color=color, linestyle=ls, linewidth=lw,
                    label=label, where="post", zorder=zo, alpha=0.92)

    n_tol = len(tol_levels)
    with _rc():
        fig, axes = plt.subplots(2, n_tol,
                                 figsize=(_COL2_W, 4.8),
                                 gridspec_kw={"hspace": 0.60, "wspace": 0.38})

        for ci, tol in enumerate(tol_levels):
            tol_str = rf"$\varepsilon={tol:.0e}$"
            T_raw = _build_matrix(tol)
            T_mw  = _build_matrix_mw(tol)

            _plot_perf_profile(axes[0, ci], T_raw,
                               f"Performance profile  {tol_str}")
            _plot_data_profile(axes[1, ci], T_mw,
                               f"Data profile  {tol_str}")

        handles_global, labels_global = [], []
        for row_axes in axes:
            for ax in row_axes:
                for h in ax.get_lines():
                    lbl = h.get_label()
                    if lbl and not lbl.startswith("_") and lbl not in labels_global:
                        handles_global.append(h)
                        labels_global.append(lbl)

        if handles_global:
            fig.legend(handles_global, labels_global,
                       loc="lower center",
                       ncol=min(len(handles_global), 5),
                       bbox_to_anchor=(0.5, -0.08),
                       frameon=True,
                       handlelength=_LEG_HLEN,
                       columnspacing=_LEG_COLSPC,
                       labelspacing=0.25,
                       fontsize=_LEG_FONT)

        plt.tight_layout(rect=[0, 0.13, 1, 1], pad=0.4)
        base = os.path.join(fig_dir, "perf_profiles_panel")
        _save(fig, base)
        plt.close(fig)


def fig_perf_profiles_comm_panel(all_logs: dict, alg_bank: list,
                                  results_dir: str,
                                  tol_levels: list = None,
                                  all_individual: dict = None) -> None:
    """
    6-panel figure (2 rows x 3 columns) using **commCost (MB)** as metric:
      Row 0 -- Performance profiles (Dolan-More):
               rho_s(tau) vs tau (comm-cost ratio)
      Row 1 -- Data profiles:
               d_s(kappa) vs kappa (comm cost in MB)

    Criterion: relF < tol.
    Saved to  results_dir/paper/perf_profiles/perf_profiles_comm_panel.{pdf,png}.
    """
    if tol_levels is None:
        tol_levels = [1e-3, 1e-6, 1e-9]

    fig_dir = os.path.join(results_dir, "paper", "perf_profiles")
    os.makedirs(fig_dir, exist_ok=True)

    alg_names = [an for an, _ in alg_bank]
    obj_names = list(all_logs.keys())
    if not obj_names or not alg_names:
        return

    groups = get_alg_groups()

    use_individual = (all_individual is not None
                      and len(all_individual) > 0)
    if use_individual:
        instance_ids = []
        for on in obj_names:
            runs = all_individual.get(on, [])
            for ri in range(len(runs)):
                instance_ids.append((on, ri))
        n_p = len(instance_ids)
    else:
        n_p = len(obj_names)

    if n_p == 0:
        return

    def _first_hit_comm(log, tol):
        """Return cumulative commCost (MB) at first step where relF < tol, or NaN."""
        relF_arr = np.asarray(log.get("relF", []), dtype=float)
        comm_arr = np.asarray(log.get("commCost", []), dtype=float)
        if len(relF_arr) == 0 or len(comm_arr) == 0:
            return np.nan
        K = min(len(relF_arr), len(comm_arr))
        hit = np.where(relF_arr[:K] < tol)[0]
        if len(hit) == 0:
            return np.nan
        return float(comm_arr[hit[0]])

    def _build_matrix(tol):
        T = np.full((len(alg_names), n_p), np.nan)
        if use_individual:
            for ai, an in enumerate(alg_names):
                for pi, (on, ri) in enumerate(instance_ids):
                    lg = all_individual[on][ri].get(an, {})
                    if lg.get("__failed__"):
                        continue
                    T[ai, pi] = _first_hit_comm(lg, tol)
        else:
            for ai, an in enumerate(alg_names):
                for pi, on in enumerate(obj_names):
                    lg = all_logs[on].get(an, {})
                    if lg.get("__failed__"):
                        continue
                    T[ai, pi] = _first_hit_comm(lg, tol)
        return T

    def _plot_perf_profile(ax, T, title_str):
        n_inst = T.shape[1]
        with np.errstate(all="ignore"):
            t_best = np.nanmin(T, axis=0)
            t_best[t_best <= 0] = np.nan
            R = T / t_best[np.newaxis, :]

        finite_R = R[np.isfinite(R)]
        if len(finite_R) == 0:
            ax.set_visible(False)
            return
        tau_max = min(float(np.nanmax(finite_R)) * 1.05, 50.0)
        tau = np.linspace(1.0, tau_max, 500)

        ax.set_xlim(1.0, tau_max)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, which="both", ls="--", alpha=_GRID_ALPHA)
        ax.set_xlabel("Communication ratio", fontsize=7.5)
        ax.set_ylabel("Fraction solved", fontsize=7.5)
        ax.set_title(title_str, pad=_TITLE_PAD, fontsize=8)

        for ai, an in enumerate(alg_names):
            r_a   = R[ai, :]
            valid = r_a[np.isfinite(r_a)]
            rho   = np.array([float(np.sum(valid <= t)) / float(n_inst)
                              for t in tau])
            label, color, ls = get_alg_style(an)
            lw = _LW_OURS if an in groups["ours"] else _LW_BASELINE
            zo = 3 if an in groups["ours"] else 2
            ax.step(tau, rho, color=color, linestyle=ls, linewidth=lw,
                    label=label, where="post", zorder=zo, alpha=0.92)

    def _plot_data_profile(ax, T, title_str):
        n_inst = T.shape[1]
        finite = T[np.isfinite(T)]
        if len(finite) == 0:
            ax.set_visible(False)
            return
        kappa_max = float(np.nanmax(finite)) * 1.05
        kappa = np.linspace(0, kappa_max, 500)

        ax.set_xlim(0, kappa_max)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, which="both", ls="--", alpha=_GRID_ALPHA)
        ax.set_xlabel("Communication budget (MB)", fontsize=7.5)
        ax.set_ylabel("Fraction solved", fontsize=7.5)
        ax.set_title(title_str, pad=_TITLE_PAD, fontsize=8)

        for ai, an in enumerate(alg_names):
            t_a = T[ai, :]
            rho = np.array([float(np.sum(t_a[np.isfinite(t_a)] <= k)) / float(n_inst)
                            for k in kappa])
            label, color, ls = get_alg_style(an)
            lw = _LW_OURS if an in groups["ours"] else _LW_BASELINE
            zo = 3 if an in groups["ours"] else 2
            ax.step(kappa, rho, color=color, linestyle=ls, linewidth=lw,
                    label=label, where="post", zorder=zo, alpha=0.92)

    n_tol = len(tol_levels)
    with _rc():
        fig, axes = plt.subplots(2, n_tol,
                                 figsize=(_COL2_W, 4.8),
                                 gridspec_kw={"hspace": 0.60, "wspace": 0.38})

        handles_global, labels_global = [], []

        for ci, tol in enumerate(tol_levels):
            tol_str = rf"$\varepsilon={tol:.0e}$"
            T = _build_matrix(tol)

            _plot_perf_profile(axes[0, ci], T,
                               f"Perf. profile (comm)  {tol_str}")

            _plot_data_profile(axes[1, ci], T,
                               f"Data profile (comm)  {tol_str}")

        for row_axes in axes:
            for ax in row_axes:
                for h in ax.get_lines():
                    lbl = h.get_label()
                    if lbl and not lbl.startswith("_") and lbl not in labels_global:
                        handles_global.append(h)
                        labels_global.append(lbl)

        if handles_global:
            fig.legend(handles_global, labels_global,
                       loc="lower center",
                       ncol=min(len(handles_global), 5),
                       bbox_to_anchor=(0.5, -0.08),
                       frameon=True,
                       handlelength=_LEG_HLEN,
                       columnspacing=_LEG_COLSPC,
                       labelspacing=0.25,
                       fontsize=_LEG_FONT)

        plt.tight_layout(rect=[0, 0.13, 1, 1], pad=0.4)
        base = os.path.join(fig_dir, "perf_profiles_comm_panel")
        _save(fig, base)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Topology robustness: grouped bar chart
# ─────────────────────────────────────────────────────────────────────────────

def fig_plot_topology_bar(topo_data: dict, results_dir: str,
                           title: str = "Topology robustness") -> None:
    """
    Grouped bar chart comparing success rate (%) across network topologies.

    Parameters
    ----------
    topo_data : {topology_name: {alg_name: success_rate_percent}}
    results_dir : output directory
    title : figure title
    """
    fig_dir = os.path.join(results_dir, "supplement", "heatmaps")
    os.makedirs(fig_dir, exist_ok=True)

    topo_names = list(topo_data.keys())
    if not topo_names:
        return
    alg_names = list(topo_data[topo_names[0]].keys())
    n_topo = len(topo_names)
    n_alg  = len(alg_names)
    if n_alg == 0:
        return

    # Colour palette for topologies (Set2 is colourblind-friendly)
    topo_palette = plt.cm.Set2(np.linspace(0.0, 0.85, n_topo))
    bar_w = 0.75 / max(n_topo, 1)
    x = np.arange(n_alg)

    with _rc():
        fig_w = max(4.5, 0.65 * n_alg + 1.0)
        fig, ax = plt.subplots(figsize=(fig_w, 3.0))

        for ti, topo_name in enumerate(topo_names):
            vals = [topo_data[topo_name].get(a, 0.0) for a in alg_names]
            offset = (ti - (n_topo - 1) / 2.0) * bar_w
            bars = ax.bar(x + offset, vals, bar_w * 0.88,
                          label=topo_name.capitalize(),
                          color=topo_palette[ti],
                          edgecolor="white", linewidth=0.5,
                          alpha=0.88, zorder=3)
            for bar, val in zip(bars, vals):
                if val > 2:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            min(val + 1.5, 103),
                            f"{val:.0f}",
                            ha="center", va="bottom", fontsize=5.5,
                            color="0.25")

        ax.set_xticks(x)
        ax.set_xticklabels([get_alg_style(a)[0] for a in alg_names],
                           rotation=40, ha="right", fontsize=7)
        ax.set_ylim(0, 112)
        ax.set_ylabel("Success rate (%)")
        ax.set_title(title, pad=4)
        ax.axhline(100, color="0.5", ls=":", lw=0.8)
        ax.grid(axis="y", ls="--", alpha=0.35)
        ax.legend(loc="upper right", frameon=True, fontsize=7,
                  handlelength=1.4, columnspacing=0.5,
                  ncol=min(n_topo, 3))

        plt.tight_layout(pad=0.5)
        base = os.path.join(fig_dir, "topology_bar")
        _save(fig, base)
        plt.close(fig)


def fig_plot_topology_heatmap(topo_steps: dict, results_dir: str) -> None:
    """
    Heatmap: rows = algorithms, cols = topologies, values = avg steps (NaN = failed).
    Colour indicates relative speed; annotated with actual values.

    Parameters
    ----------
    topo_steps : {topology_name: {alg_name: avg_steps_or_nan}}
    results_dir : output directory
    """
    fig_dir = os.path.join(results_dir, "supplement", "heatmaps")
    os.makedirs(fig_dir, exist_ok=True)

    if not topo_steps:
        return

    topo_names = list(topo_steps.keys())
    alg_names  = list(next(iter(topo_steps.values())).keys())

    data = np.full((len(alg_names), len(topo_names)), np.nan)
    for tj, tname in enumerate(topo_names):
        for ai, aname in enumerate(alg_names):
            data[ai, tj] = topo_steps[tname].get(aname, np.nan)

    labels_alg  = [get_alg_style(a)[0] for a in alg_names]
    labels_topo = [t.capitalize() for t in topo_names]

    with np.errstate(all="ignore"):
        col_min = np.nanmin(data, axis=0, keepdims=True)
        col_max = np.nanmax(data, axis=0, keepdims=True)
        norm_data = (data - col_min) / np.maximum(col_max - col_min, 1e-9)

    with _rc():
        fig_h = max(2.6, 0.38 * len(alg_names) + 0.9)
        fig_w = max(4.0, 0.9 * len(topo_names) + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(norm_data, aspect="auto", cmap="YlOrBr",
                       vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_xticks(range(len(topo_names)))
        ax.set_xticklabels(labels_topo, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(alg_names)))
        ax.set_yticklabels(labels_alg, fontsize=8)

        for ai in range(len(alg_names)):
            for tj in range(len(topo_names)):
                v = data[ai, tj]
                if np.isfinite(v):
                    txt = f"{int(v)}" if v < 1e4 else f"{v:.1e}"
                    brightness = float(norm_data[ai, tj])
                    color = "white" if brightness < 0.25 or brightness > 0.75 else "black"
                    ax.text(tj, ai, txt, ha="center", va="center",
                            fontsize=6.5, color=color, fontweight="bold")

        cb = plt.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label("Relative steps (low = fast)", fontsize=7)
        ax.set_title("Steps to convergence by topology", pad=5)
        plt.tight_layout(pad=0.4)

        base = os.path.join(fig_dir, "topology_heatmap")
        _save(fig, base)
        plt.close(fig)


def fig_topology_combined_panel(topo_data: dict, topo_steps: dict,
                                 results_dir: str,
                                 title: str = "Topology robustness") -> None:
    """
    Combined topology figure: bar chart (top) + heatmap (bottom) in a single
    PDF / PNG file.  Replaces calling fig_plot_topology_bar and
    fig_plot_topology_heatmap separately.

    Parameters
    ----------
    topo_data  : {topology_name: {alg_name: success_rate_pct}}
    topo_steps : {topology_name: {alg_name: avg_steps_or_nan}}
    results_dir : output root directory
    title : super-title
    """
    fig_dir = os.path.join(results_dir, "paper", "topology_combined")
    os.makedirs(fig_dir, exist_ok=True)

    topo_names = list(topo_data.keys())
    if not topo_names:
        return
    alg_names = list(topo_data[topo_names[0]].keys())
    n_topo = len(topo_names)
    n_alg  = len(alg_names)
    if n_alg == 0:
        return

    # ── Build colour/normalised data for heatmap ──────────────────────────
    hmap_data = np.full((n_alg, n_topo), np.nan)
    for tj, tname in enumerate(topo_names):
        for ai, aname in enumerate(alg_names):
            hmap_data[ai, tj] = topo_steps.get(tname, {}).get(aname, np.nan)

    with np.errstate(all="ignore"):
        col_min  = np.nanmin(hmap_data, axis=0, keepdims=True)
        col_max  = np.nanmax(hmap_data, axis=0, keepdims=True)
        norm_hmap = (hmap_data - col_min) / np.maximum(col_max - col_min, 1e-9)

    # ── Layout ────────────────────────────────────────────────────────────
    topo_palette = plt.cm.Set2(np.linspace(0.0, 0.85, n_topo))
    bar_w = 0.75 / max(n_topo, 1)
    x = np.arange(n_alg)

    fig_w   = max(5.0, 0.65 * n_alg + 1.5)
    bar_h   = 2.8
    hmap_h  = max(2.2, 0.34 * n_alg + 0.9)
    fig_h   = bar_h + hmap_h + 0.5

    labels_alg  = [get_alg_style(a)[0] for a in alg_names]
    labels_topo = [t.capitalize() for t in topo_names]

    with _rc():
        fig, (ax_bar, ax_hmap) = plt.subplots(
            2, 1, figsize=(fig_w, fig_h),
            gridspec_kw={"hspace": 0.55,
                         "height_ratios": [bar_h, hmap_h]})

        # ── Bar chart ────────────────────────────────────────────────────
        for ti, topo_name in enumerate(topo_names):
            vals   = [topo_data[topo_name].get(a, 0.0) for a in alg_names]
            offset = (ti - (n_topo - 1) / 2.0) * bar_w
            bars = ax_bar.bar(x + offset, vals, bar_w * 0.88,
                              label=topo_name.capitalize(),
                              color=topo_palette[ti],
                              edgecolor="white", linewidth=0.5,
                              alpha=0.88, zorder=3)
            for bar, val in zip(bars, vals):
                if val > 2:
                    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                                min(val + 1.5, 103),
                                f"{val:.0f}",
                                ha="center", va="bottom", fontsize=5.5,
                                color="0.25")

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels_alg, rotation=40, ha="right", fontsize=7)
        ax_bar.set_ylim(0, 112)
        ax_bar.set_ylabel("Success rate (%)")
        ax_bar.set_title("Success rate by topology", pad=4)
        ax_bar.axhline(100, color="0.5", ls=":", lw=0.8)
        ax_bar.grid(axis="y", ls="--", alpha=0.35)
        ax_bar.legend(loc="upper right", frameon=True, fontsize=7,
                      handlelength=1.4, ncol=min(n_topo, 3))

        # ── Heatmap ──────────────────────────────────────────────────────
        im = ax_hmap.imshow(norm_hmap, aspect="auto", cmap="YlOrBr",
                             vmin=0.0, vmax=1.0, interpolation="nearest")
        ax_hmap.set_xticks(range(n_topo))
        ax_hmap.set_xticklabels(labels_topo, rotation=30, ha="right", fontsize=7.5)
        ax_hmap.set_yticks(range(n_alg))
        ax_hmap.set_yticklabels(labels_alg, fontsize=7.5)
        ax_hmap.set_title("Steps to convergence by topology", pad=4)

        for ai in range(n_alg):
            for tj in range(n_topo):
                v = hmap_data[ai, tj]
                if np.isfinite(v):
                    txt = f"{int(v)}" if v < 1e4 else f"{v:.1e}"
                    brightness = float(norm_hmap[ai, tj])
                    color = "white" if brightness < 0.25 or brightness > 0.75 else "black"
                    ax_hmap.text(tj, ai, txt, ha="center", va="center",
                                 fontsize=6.0, color=color, fontweight="bold")

        fig.colorbar(im, ax=ax_hmap, shrink=0.8,
                     label="Relative steps (low = fast)")

        fig.suptitle(title, fontsize=9, y=0.995)
        plt.tight_layout(pad=0.4)

        base = os.path.join(fig_dir, "topology_combined")
        _save(fig, base)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Communication vs Steps efficiency scatter (Pareto frontier)
# ─────────────────────────────────────────────────────────────────────────────

def fig_plot_comm_budget(comm_steps_data: dict,
                          results_dir: str,
                          obj_name: str = "") -> None:
    """
    Log-log scatter plot of communication cost vs iteration count.

    Each algorithm occupies one point; algorithms on or near the Pareto
    frontier (low comm AND low steps) are highlighted with a bold border.

    Parameters
    ----------
    comm_steps_data : {alg_name: {"comm": float, "steps": int}}
    results_dir     : output directory
    obj_name        : objective name (for title / filename)
    """
    fig_dir = os.path.join(results_dir, "supplement", "pareto")
    os.makedirs(fig_dir, exist_ok=True)

    groups = get_alg_groups()

    pts_alg, pts_comm, pts_steps = [], [], []
    for alg_name, d in comm_steps_data.items():
        c = float(d.get("comm",  np.nan))
        s = float(d.get("steps", np.nan))
        if np.isfinite(c) and np.isfinite(s) and c > 0 and s > 0:
            pts_alg.append(alg_name)
            pts_comm.append(c)
            pts_steps.append(s)

    if not pts_alg:
        return

    # Identify Pareto-optimal points (minimise both axes)
    pareto_mask = []
    for i, (ci, si) in enumerate(zip(pts_comm, pts_steps)):
        dominated = any(
            (pts_comm[j] <= ci and pts_steps[j] <= si and
             (pts_comm[j] < ci or pts_steps[j] < si))
            for j in range(len(pts_alg)) if j != i
        )
        pareto_mask.append(not dominated)

    # Sort Pareto front for drawing
    if any(pareto_mask):
        pf_pts = sorted(
            [(pts_comm[i], pts_steps[i])
             for i in range(len(pts_alg)) if pareto_mask[i]],
            key=lambda t: t[0]
        )
        pf_c = [t[0] for t in pf_pts]
        pf_s = [t[1] for t in pf_pts]

    with _rc():
        fig, ax = plt.subplots(figsize=(_COL1_W, _CELL_H))
        ax.grid(True, which="both", ls="--", alpha=_GRID_ALPHA)
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Draw Pareto frontier line
        if any(pareto_mask) and len(pf_c) > 1:
            ax.step(pf_c, pf_s, where="post", color="gold",
                    linewidth=1.0, linestyle="--", zorder=1,
                    label="Pareto front", alpha=0.7)

        for i, alg_name in enumerate(pts_alg):
            label, color, _ = get_alg_style(alg_name)
            is_ours   = alg_name in groups["ours"]
            is_pareto = pareto_mask[i]
            marker = "^" if is_ours else "o"
            ms     = 8  if is_ours else 6
            ew     = 1.6 if is_pareto else 0.3
            ec     = "black" if is_pareto else "white"
            ax.scatter(pts_comm[i], pts_steps[i],
                       c=[color], s=ms ** 2,
                       marker=marker, edgecolors=ec, linewidths=ew,
                       label=label, zorder=4, alpha=0.93)
            ax.annotate(label,
                        (pts_comm[i], pts_steps[i]),
                        fontsize=5.5, ha="left", va="bottom",
                        xytext=(3, 2), textcoords="offset points",
                        color=color)

        ax.set_xlabel("Communication cost (MB)")
        ax.set_ylabel("Steps to convergence")
        t = "Comm-iteration efficiency"
        if obj_name:
            t += f"  [{_clean_title(obj_name)}]"
        ax.set_title(t, pad=_TITLE_PAD)
        # Suppress duplicate labels in legend
        handles, labels_l = ax.get_legend_handles_labels()
        seen = set()
        h_dedup, l_dedup = [], []
        for h, lbl in zip(handles, labels_l):
            if lbl not in seen:
                seen.add(lbl); h_dedup.append(h); l_dedup.append(lbl)
        ax.legend(h_dedup, l_dedup, loc="upper right", frameon=True,
                  ncol=2, fontsize=6, handlelength=1.2)
        plt.tight_layout(pad=0.4)

        suf = obj_name.replace(" ", "_") if obj_name else "all"
        base = os.path.join(fig_dir, f"comm_budget_{suf}")
        _save(fig, base)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Convergence rate bar chart (estimated from log-combo slope)
# ─────────────────────────────────────────────────────────────────────────────

def fig_plot_convergence_rate(log_merged: dict, alg_bank: list,
                               results_dir: str,
                               obj_name: str = "") -> None:
    """
    Estimate per-algorithm linear convergence rate and visualise as a bar chart.

    The rate is estimated as the slope of log10(combo) vs iteration
    in the final 40 % of the convergence curve (linear-convergence regime).
    A more negative bar = faster convergence.

    Parameters
    ----------
    log_merged : {alg_name: log_dict}
    alg_bank   : list of (name, callable)
    results_dir : output directory
    obj_name   : objective name (for title / filename)
    """
    fig_dir = os.path.join(results_dir, "supplement", "rates")
    os.makedirs(fig_dir, exist_ok=True)

    groups = get_alg_groups()
    rates  = {}

    for alg_name, _ in alg_bank:
        if alg_name not in log_merged:
            continue
        combo = np.asarray(log_merged[alg_name].get("combo", [np.nan])).ravel()
        valid = np.isfinite(combo) & (combo > 1e-14)
        if valid.sum() < 6:
            continue
        combo_v = combo[valid]
        k_v     = np.where(valid)[0].astype(float)
        n       = len(k_v)
        start   = max(0, n * 6 // 10)
        if n - start < 3:
            start = max(0, n - 5)
        kf = k_v[start:]
        yf = np.log10(combo_v[start:])
        if len(kf) < 2:
            continue
        A_mat = np.column_stack([kf, np.ones(len(kf))])
        slope = float(np.linalg.lstsq(A_mat, yf, rcond=None)[0][0])
        rates[alg_name] = slope

    if not rates:
        return

    alg_order = [a for a, _ in alg_bank if a in rates]
    vals    = [rates[a]           for a in alg_order]
    colors  = [get_alg_style(a)[1] for a in alg_order]
    labels  = [get_alg_style(a)[0] for a in alg_order]

    with _rc():
        fig_w = max(_COL1_W, 0.60 * len(alg_order) + 0.8)
        fig, ax = plt.subplots(figsize=(fig_w, _CELL_H))

        xpos = np.arange(len(alg_order))
        bars = ax.bar(xpos, vals, color=colors, edgecolor="white",
                      linewidth=0.6, zorder=3, alpha=0.88)

        # Gold border on best (most negative = fastest)
        best_idx = int(np.argmin(vals))
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2.0)
        bars[best_idx].set_alpha(1.0)

        # Annotate bars
        for bar, v in zip(bars, vals):
            yoff = v * 0.04
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v - yoff,
                    f"{v:.3f}",
                    ha="center",
                    va="top" if v < 0 else "bottom",
                    fontsize=6.0, color="white" if abs(v) > 0.03 else "0.2",
                    fontweight="bold")

        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel(r"$\log_{10}$ per iteration")
        t = "Estimated convergence rate"
        if obj_name:
            t += f"  [{_clean_title(obj_name)}]"
        ax.set_title(t, pad=_TITLE_PAD)
        ax.axhline(0, color="0.4", ls=":", lw=0.8)
        ax.grid(axis="y", ls="--", alpha=_GRID_ALPHA)

        plt.tight_layout(pad=0.4)
        suf = obj_name.replace(" ", "_") if obj_name else "all"
        base = os.path.join(fig_dir, f"conv_rate_{suf}")
        _save(fig, base)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  2 x 3 Scalability panel figure
# ─────────────────────────────────────────────────────────────────────────────

def fig_plot_scalability_panel(data_N: dict, data_d: dict,
                                N_vals: list, d_vals: list,
                                results_dir: str,
                                obj_name: str = "ridge") -> None:
    """
    Publication-quality 2 x 3 panel figure for scalability experiments.

    Row 0 - vary number of agents N
    Row 1 - vary problem dimension d
    Columns: steps to convergence | wall-clock time (s) | communication (MB)

    Parameters
    ----------
    data_N  : {"steps": {alg: list}, "time": ..., "comm": ...}
    data_d  : same structure, for dimension sweep
    N_vals  : list of N values (x-axis row 0)
    d_vals  : list of d values (x-axis row 1)
    results_dir : output directory
    obj_name : objective name (title + filename)
    """
    fig_dir = os.path.join(results_dir, "paper", "scalability_panel")
    os.makedirs(fig_dir, exist_ok=True)

    groups = get_alg_groups()
    metric_keys = ["steps", "time", "comm"]
    ylabels = {
        "steps": "Steps to convergence",
        "time":  "Wall-clock time (s)",
        "comm":  "Communication (MB)",
    }
    row_info = [
        (data_N, N_vals, r"Agents  $N$"),
        (data_d, d_vals, r"Dimension  $d$"),
    ]
    row_titles = [
        f"Scalability in $N$ - {_clean_title(obj_name)}",
        f"Scalability in $d$ - {_clean_title(obj_name)}",
    ]

    with _rc():
        fig, axes = plt.subplots(2, 3, figsize=(_COL2_W, 4.0),
                                 gridspec_kw={"hspace": 0.55, "wspace": 0.40})

        handles_global, labels_global = [], []

        for row, (row_data, x_vals, x_label) in enumerate(row_info):
            for col, metric in enumerate(metric_keys):
                ax = axes[row, col]
                ax.grid(True, which="both", ls="--", alpha=_GRID_ALPHA)
                ax.set_yscale("log")
                ax.tick_params(which="both", direction="in", top=False, right=False)

                alg_data = row_data.get(metric, {})
                for alg_name, val_list in alg_data.items():
                    if not val_list:
                        continue
                    y = np.array([v if (v is not None and np.isfinite(v))
                                  else np.nan for v in val_list], dtype=float)
                    if np.all(np.isnan(y)):
                        continue
                    label, color, ls = get_alg_style(alg_name)
                    lw = _LW_OURS if alg_name in groups["ours"] else _LW_BASELINE
                    zo = 3   if alg_name in groups["ours"] else 2
                    h, = ax.plot(x_vals, y, "o-",
                                 color=color, linestyle=ls,
                                 linewidth=lw, markersize=3.5,
                                 label=label, zorder=zo, alpha=0.92)
                    if label not in labels_global:
                        handles_global.append(h)
                        labels_global.append(label)

                ax.set_xticks(x_vals)
                ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
                ax.set_xlabel(x_label, fontsize=8)
                ax.set_ylabel(ylabels[metric], fontsize=8)
                if col == 1:
                    ax.set_title(row_titles[row], pad=4)

        # Shared legend below
        if handles_global:
            ncols = min(len(handles_global), 7)
            fig.legend(handles_global, labels_global,
                       loc="lower center", ncol=ncols,
                       bbox_to_anchor=(0.5, -0.01),
                       frameon=True, handlelength=_LEG_HLEN,
                       columnspacing=_LEG_COLSPC, labelspacing=0.25,
                       fontsize=_LEG_FONT)

        plt.tight_layout(rect=[0, 0.07, 1, 1])
        base = os.path.join(fig_dir, f"scalability_panel_{obj_name}")
        _save(fig, base)
        plt.close(fig)


def fig_scalability_multiobj_panel(all_scale_data: dict,
                                    N_vals: list, d_vals: list,
                                    results_dir: str,
                                    metrics: list = None) -> None:
    """
    Multi-function scalability panel.

    Layout: rows = objectives, columns = (N-steps, d-steps, N-comm, d-comm).
    Each row shows one objective's scalability behaviour.

    Parameters
    ----------
    all_scale_data : {obj_name: {"data_N": {metric: {alg: vals}},
                                  "data_d": {metric: {alg: vals}}}}
    N_vals         : list of agent counts (x-axis for vary-N columns)
    d_vals         : list of dimensions   (x-axis for vary-d columns)
    results_dir    : output root directory
    metrics        : which metrics to show; default ["steps", "comm"]
    """
    if metrics is None:
        metrics = ["steps", "comm"]

    fig_dir = os.path.join(results_dir, "paper", "scalability_panel")
    os.makedirs(fig_dir, exist_ok=True)

    obj_names  = list(all_scale_data.keys())
    n_obj      = len(obj_names)
    if n_obj == 0:
        return

    groups = get_alg_groups()

    ylabels = {
        "steps": "Steps",
        "time":  "Time (s)",
        "comm":  "Comm (MB)",
    }
    x_infos = [
        ("data_N", N_vals, r"$N$"),
        ("data_d", d_vals, r"$d$"),
    ]

    n_cols = len(x_infos) * len(metrics)
    cell_w, cell_h = 2.6, 2.2
    fig_w = cell_w * n_cols + 0.6
    fig_h = cell_h * n_obj + 0.6

    col_titles = [f"{'N' if ki == 'data_N' else 'd'} vs {m}"
                  for ki, _, _ in x_infos for m in metrics]

    with _rc():
        fig, axes = plt.subplots(n_obj, n_cols,
                                 figsize=(fig_w, fig_h),
                                 squeeze=False,
                                 gridspec_kw={"hspace": 0.45, "wspace": 0.40})

        handles_global, labels_global = [], []

        for ri, obj_name in enumerate(obj_names):
            sc = all_scale_data[obj_name]
            ci = 0
            for key, x_vals, x_lbl in x_infos:
                row_data = sc.get(key, {})
                for metric in metrics:
                    ax = axes[ri, ci]
                    ax.grid(True, which="both", ls="--", alpha=_GRID_ALPHA)
                    ax.set_yscale("log")

                    alg_data = row_data.get(metric, {})
                    for alg_name, val_list in alg_data.items():
                        if not val_list:
                            continue
                        y = np.array([v if (v is not None and np.isfinite(v))
                                      else np.nan for v in val_list], dtype=float)
                        if np.all(np.isnan(y)):
                            continue
                        label, color, ls = get_alg_style(alg_name)
                        lw = _LW_OURS if alg_name in groups["ours"] else _LW_BASELINE
                        zo = 3   if alg_name in groups["ours"] else 2
                        h, = ax.plot(x_vals, y, "o-",
                                     color=color, linestyle=ls,
                                     linewidth=lw, markersize=3.0,
                                     label=label, zorder=zo, alpha=0.90)
                        if label not in labels_global:
                            handles_global.append(h)
                            labels_global.append(label)

                    ax.set_xticks(x_vals)
                    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
                    ax.tick_params(labelsize=6)
                    if ri == n_obj - 1:
                        ax.set_xlabel(x_lbl, fontsize=7)
                    if ci == 0:
                        ax.set_ylabel(_clean_title(obj_name, short=True), fontsize=7)
                    elif ci % len(metrics) == 0:
                        ax.set_ylabel(ylabels.get(metric, metric), fontsize=7)
                    if ri == 0:
                        ax.set_title(col_titles[ci], pad=3, fontsize=7)
                    ci += 1

        if handles_global:
            n_h = len(handles_global)
            fig.legend(handles_global, labels_global,
                       loc="lower center",
                       ncol=min(n_h, 6),
                       bbox_to_anchor=(0.5, 0.0),
                       frameon=True, handlelength=_LEG_HLEN,
                       columnspacing=_LEG_COLSPC, labelspacing=0.25,
                       fontsize=_LEG_FONT)

        plt.tight_layout(rect=[0, 0.06, 1, 1], pad=0.3)
        base = os.path.join(fig_dir, "scalability_multiobj_panel")
        _save(fig, base)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  3-panel convergence trio (steps / comm / time vs combo)
# ─────────────────────────────────────────────────────────────────────────────

def fig_plot_convergence_trio(log_merged: dict, alg_bank: list,
                               results_dir: str,
                               obj_name: str = "") -> None:
    """
    Side-by-side triple panel for one objective:
      Left  - iterations vs combo
      Centre - comm cost (MB) vs combo
      Right  - wall-clock time (s) vs combo

    Allows direct comparison of three efficiency axes in a single figure.

    Parameters
    ----------
    log_merged : merged log dict {alg_name: log}
    alg_bank   : list of (name, callable) in display order
    results_dir : output directory
    obj_name   : objective function name (for title / filename)
    """
    fig_dir = os.path.join(results_dir, "paper", "trio")
    os.makedirs(fig_dir, exist_ok=True)

    x_configs = [
        ("steps",    "Iteration"),
        ("commCost", "Communication (MB)"),
        ("timeCost", "Time (s)"),
    ]
    y_key = "combo"

    groups = get_alg_groups()

    with _rc():
        # Use constrained_layout to avoid tight_layout warnings with figure legend
        fig, axes = plt.subplots(1, 3, figsize=(_COL2_W, 2.4),
                                 gridspec_kw={"wspace": 0.45},
                                 constrained_layout=False)

        handles_global, labels_global = [], []

        for ax, (x_key, x_label) in zip(axes, x_configs):
            _format_ax(ax, "semilogy")
            ax.set_xlabel(x_label, fontsize=8)
            ax.set_ylabel(r"$\mathrm{combo}_k$", fontsize=8)

            for alg_name, _ in alg_bank:
                if alg_name not in log_merged:
                    continue
                L = log_merged[alg_name]
                is_failed = bool(L.get("__failed__"))
                y = _get_field(L, y_key)
                if x_key.lower() == "steps":
                    x = (np.arange(1, len(y) + 1, dtype=float)
                         if y is not None else None)
                else:
                    x = _get_field(L, x_key)
                x, y = _cleanup_xy(x, y, y_key)
                if x is None:
                    continue
                label, color, ls = get_alg_style(alg_name)
                if is_failed:
                    label_use = label + r" $\times$"
                    if label_use not in labels_global:
                        h, = ax.plot([], [], color=(0.65, 0.65, 0.65),
                                     linestyle="none", marker="x", markersize=6,
                                     markeredgewidth=1.5, label=label_use, alpha=0.85)
                        handles_global.append(h)
                        labels_global.append(label_use)
                else:
                    label_use = label
                    lw = _LW_OURS if alg_name in groups["ours"] else _LW_BASELINE
                    zo = 3   if alg_name in groups["ours"] else 2
                    mk = _get_marker_props(alg_name, len(x))
                    h, = ax.plot(x, y, color=color, linestyle=ls,
                                 linewidth=lw, label=label_use, zorder=zo,
                                 alpha=0.92, **mk)

                    y_q25 = _get_field(L, y_key + "_q25")
                    y_q75 = _get_field(L, y_key + "_q75")
                    if y_q25 is not None and y_q75 is not None:
                        K_band = min(len(x), len(y_q25), len(y_q75))
                        q25 = np.clip(y_q25[:K_band], _FLOOR, None)
                        q75 = np.clip(y_q75[:K_band], _FLOOR, None)
                        ax.fill_between(x[:K_band], q25, q75,
                                        color=color, alpha=0.15, zorder=zo - 1,
                                        linewidth=0)

                    if label_use not in labels_global:
                        handles_global.append(h)
                        labels_global.append(label_use)

        for ax in axes:
            _add_tol_line(ax, "combo")
        axes[1].set_title(_clean_title(obj_name) if obj_name else "Convergence", pad=_TITLE_PAD)

        fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.12,
                            wspace=0.40)
        if handles_global:
            _add_shared_legend(fig, handles_global, labels_global)
        suf = obj_name.replace(" ", "_") if obj_name else "all"
        base = os.path.join(fig_dir, f"trio_{suf}")
        _save(fig, base)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  CE benefit figure
#  Shows: for each function, compare DisGrem vs CeDisGrem and
#         AdaDisGrem vs CeAdaDisGrem on iterations and communication cost.
# ─────────────────────────────────────────────────────────────────────────────

def fig_plot_ce_benefit(all_logs: dict, results_dir: str) -> None:
    """
    Grouped bar chart illustrating the benefit of consensus-efficient (CE) variants.

    Two panels: left = iterations to convergence, right = total communication (MB).
    X-axis: function names.  Four bars per function: DisGrem, CeDisGrem,
    AdaDisGrem, CeAdaDisGrem.  CE variants are hatched to distinguish them.

    Saved to results_dir/paper/ce_benefit/ce_benefit_{metric}.{pdf,png}.
    """
    fig_dir = os.path.join(results_dir, "paper", "ce_benefit")
    os.makedirs(fig_dir, exist_ok=True)

    alg_order = ["DisGrem", "CeDisGrem", "AdaDisGrem", "CeAdaDisGrem"]
    obj_names = list(all_logs.keys())
    n_obj = len(obj_names)
    if n_obj == 0:
        return

    ce_colors = [get_alg_style(a)[1] for a in alg_order]
    ce_labels = [get_alg_style(a)[0] for a in alg_order]
    hatches = ["", "///", "", "///"]

    for metric, ylabel, getter in [
        ("steps", "Iterations to convergence",
         lambda lg: float(len(lg.get("ValueF", [np.nan]))) if not lg.get("__failed__") else np.nan),
        ("comm",  "Communication cost (MB)",
         lambda lg: float(np.nanmax(lg.get("commCost", [np.nan]))) if not lg.get("__failed__") else np.nan),
    ]:
        n_bars = len(alg_order)
        bar_w = 0.18
        x = np.arange(n_obj)
        offsets = np.linspace(-(n_bars - 1) / 2.0, (n_bars - 1) / 2.0, n_bars) * bar_w

        with _rc():
            fig_w = max(5.5, 0.75 * n_obj + 1.5)
            fig, ax = plt.subplots(figsize=(fig_w, 3.2))

            for bi, (alg, lbl, col, hatch) in enumerate(
                    zip(alg_order, ce_labels, ce_colors, hatches)):
                vals = []
                for on in obj_names:
                    lg = all_logs[on].get(alg, {})
                    vals.append(getter(lg))
                vals = np.array(vals, dtype=float)

                # NaN → draw empty bar with 'DNF' text
                finite_max = np.nanmax(vals) if np.any(np.isfinite(vals)) else 1.0
                bar_heights = np.where(np.isfinite(vals), vals, 0.0)

                bars = ax.bar(x + offsets[bi], bar_heights, bar_w * 0.92,
                              label=lbl, color=col, hatch=hatch,
                              edgecolor="white", linewidth=0.5,
                              alpha=0.88, zorder=3)

                # Mark DNF bars
                for i, (bh, bv) in enumerate(zip(bars, vals)):
                    if not np.isfinite(bv):
                        ax.text(bh.get_x() + bh.get_width() / 2,
                                finite_max * 0.05, "DNF",
                                ha="center", va="bottom",
                                fontsize=5.5, color="0.45", rotation=90)

            obj_labels = [_clean_title(o, short=True) for o in obj_names]
            ax.set_xticks(x)
            ax.set_xticklabels(obj_labels, rotation=30, ha="right", fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_title("CE benefit: DisGrem vs CeDisGrem, AdaDisGrem vs CeAdaDisGrem",
                         pad=4, fontsize=8)
            ax.grid(True, axis="y", ls="--", alpha=0.35)
            ax.tick_params(axis="x", which="both", bottom=False)
            ax.legend(ncol=2, frameon=True, fontsize=7,
                      handlelength=1.6, columnspacing=0.8, labelspacing=0.3)
            plt.tight_layout(pad=0.5)

            base = os.path.join(fig_dir, f"ce_benefit_{metric}")
            _save(fig, base)
            plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Communication savings figure
#  Shows per-algorithm average communication cost across all tested functions,
#  with our algorithms visually highlighted vs baselines.
# ─────────────────────────────────────────────────────────────────────────────

def fig_plot_comm_savings(all_logs: dict, alg_bank: list,
                          results_dir: str) -> None:
    """
    Horizontal bar chart of average communication cost (MB) per algorithm,
    averaged over all functions in all_logs.  Our algorithms are colored with
    full saturation; baselines are grayed.  Ratio annotations show savings vs
    the cheapest first-order baseline (EXTRA or DIGing).

    Saved to results_dir/paper/comm_savings/comm_savings.{pdf,png}.
    """
    fig_dir = os.path.join(results_dir, "paper", "comm_savings")
    os.makedirs(fig_dir, exist_ok=True)

    groups = get_alg_groups()

    # Build per-algorithm average comm costs
    avg_comm = {}
    for alg_name, _ in alg_bank:
        costs = []
        for on, log_obj in all_logs.items():
            if alg_name not in log_obj:
                continue
            lg = log_obj[alg_name]
            if lg.get("__failed__"):
                continue
            c = _get_field(lg, "commCost")
            if c is not None and len(c) > 0:
                v = float(np.nanmax(c))
                if np.isfinite(v):
                    costs.append(v)
        avg_comm[alg_name] = float(np.mean(costs)) if costs else np.nan

    alg_names = [an for an, _ in alg_bank if an in avg_comm]
    if not alg_names:
        return

    # Reference: cheapest first-order baseline (EXTRA or DIGing) for ratio annotation
    ref_val = np.nan
    for ref_name in ("EXTRA", "DIGing"):
        if ref_name in avg_comm and np.isfinite(avg_comm[ref_name]):
            ref_val = avg_comm[ref_name]; break

    vals  = np.array([avg_comm[a] for a in alg_names], dtype=float)
    order = np.argsort(vals)[::-1]  # descending: longest bar at top
    alg_sorted = [alg_names[i] for i in order]
    val_sorted = vals[order]

    bar_colors = []
    alphas     = []
    for a in alg_sorted:
        _, col, _ = get_alg_style(a)
        if a in groups["ours"]:
            bar_colors.append(col)
            alphas.append(0.92)
        else:
            bar_colors.append(col)
            alphas.append(0.72)

    with _rc():
        fig_h = max(2.8, 0.32 * len(alg_sorted) + 0.8)
        fig, ax = plt.subplots(figsize=(5.5, fig_h))

        y_pos = np.arange(len(alg_sorted))
        for i, (a, v, col, alph) in enumerate(
                zip(alg_sorted, val_sorted, bar_colors, alphas)):
            if not np.isfinite(v):
                continue
            label_str, _, _ = get_alg_style(a)
            bar = ax.barh(y_pos[i], v, color=col, alpha=alph,
                          edgecolor="white", linewidth=0.5, zorder=3)

            # Ratio annotation
            if np.isfinite(ref_val) and v > 0 and a not in ("EXTRA", "DIGing"):
                ratio = ref_val / v
                if ratio > 1.0:
                    ax.text(v + 0.01 * val_sorted[np.isfinite(val_sorted)].max(),
                            y_pos[i], rf"$\times${ratio:.1f} savings",
                            va="center", ha="left", fontsize=6.5,
                            color="0.30" if a in groups["ours"] else "0.55")

        y_labels = [get_alg_style(a)[0] for a in alg_sorted]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=7.5)
        ax.set_xlabel("Average communication cost (MB)")
        ax.set_title("Communication savings vs baselines", pad=4)
        ax.grid(True, axis="x", ls="--", alpha=0.35)

        # Subtle legend patch distinguishing "ours" vs "baselines"
        from matplotlib.patches import Patch
        ours_col = get_alg_style("DisGrem")[1]
        base_col = get_alg_style("EXTRA")[1]
        legend_handles = [
            Patch(facecolor=ours_col, alpha=0.92, label="Proposed"),
            Patch(facecolor=base_col, alpha=0.72, label="Baselines"),
        ]
        ax.legend(handles=legend_handles, fontsize=_LEG_FONT, loc="lower right",
                  frameon=True, handlelength=1.2)

        plt.tight_layout(pad=0.5)
        base = os.path.join(fig_dir, "comm_savings")
        _save(fig, base)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Robustness: Success-rate heatmap table
# ─────────────────────────────────────────────────────────────────────────────

def fig_success_rate_table(success_data: dict, alg_names: list,
                           obj_names: list, scenario_name: str,
                           results_dir: str) -> None:
    """
    Render success-rate (%) as a coloured heatmap table.

    Parameters
    ----------
    success_data : {obj_name: {alg_name: float(%)}}
    alg_names    : ordered algorithm names (rows)
    obj_names    : ordered objective names (columns)
    scenario_name: 'near' / 'far' (for title & filename)
    results_dir  : output root
    """
    fig_dir = os.path.join(results_dir, "paper")
    os.makedirs(fig_dir, exist_ok=True)

    n_alg = len(alg_names)
    n_obj = len(obj_names)
    mat = np.full((n_alg, n_obj), np.nan)
    for ai, an in enumerate(alg_names):
        for oi, on in enumerate(obj_names):
            mat[ai, oi] = success_data.get(on, {}).get(an, np.nan)

    with _rc():
        fig_w = max(0.85 * n_obj + 1.8, 5.0)
        fig_h = max(0.38 * n_alg + 1.2, 3.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=100, aspect="auto")

        for ai in range(n_alg):
            for oi in range(n_obj):
                val = mat[ai, oi]
                if np.isfinite(val):
                    tc = "white" if val < 45 else "black"
                    ax.text(oi, ai, f"{val:.0f}",
                            ha="center", va="center",
                            fontsize=7.5, fontweight="bold", color=tc)

        col_labels = [_clean_title(on) for on in obj_names]
        row_labels = [get_alg_style(an)[0] for an in alg_names]

        ax.set_xticks(range(n_obj))
        ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=7.5)
        ax.set_yticks(range(n_alg))
        ax.set_yticklabels(row_labels, fontsize=7.5)

        ax.set_title(
            f"Success Rate (\\%) - {scenario_name.capitalize()}"
            if _USE_USETEX else
            f"Success Rate (%) - {scenario_name.capitalize()}",
            fontsize=9, pad=8)

        cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.04)
        sr_label = "Success Rate (\\%)" if _USE_USETEX else "Success Rate (%)"
        cbar.set_label(sr_label, fontsize=7.5)
        cbar.ax.tick_params(labelsize=7)

        plt.tight_layout()
        base = os.path.join(fig_dir, f"success_rate_{scenario_name}")
        _save(fig, base)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Robustness: Parameter-sensitivity gradient-colour sweep
# ─────────────────────────────────────────────────────────────────────────────

def fig_param_sweep_gradient(entries: list, group_label: str,
                              obj_name: str, results_dir: str,
                              cmap_name: str = "viridis") -> None:
    """
    Gradient-coloured parameter-sensitivity figure (one subplot per sweep).

    Parameters
    ----------
    entries : list of dicts, each containing
        alg_name    : str
        param_label : str  (e.g. 'alpha' or 'M')
        factors     : sorted list/array of float multipliers
        curves      : list of (factor, relF_mean_array, is_diverged_bool)
    group_label : figure super-title (e.g. "DisGrem family")
    obj_name    : function name
    results_dir : output root
    cmap_name   : sequential matplotlib colormap
    """
    fig_dir = os.path.join(results_dir, "paper", "param_sweep")
    os.makedirs(fig_dir, exist_ok=True)

    n_sub = len(entries)
    if n_sub == 0:
        return
    n_cols = min(n_sub, 2)
    n_rows = int(np.ceil(n_sub / n_cols))

    cmap = plt.colormaps[cmap_name]
    all_factors = entries[0]["factors"]
    log_lo = np.log10(float(min(all_factors)))
    log_hi = np.log10(float(max(all_factors)))
    log_span = max(log_hi - log_lo, 1e-6)

    with _rc():
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(_COL2_W, _CELL_H * n_rows + 0.6),
            squeeze=False)

        for si, entry in enumerate(entries):
            ax = axes[si // n_cols, si % n_cols]
            an = entry["alg_name"]
            pl = entry["param_label"]
            curves = entry["curves"]

            _format_ax(ax, "semilogy")
            label_alg = get_alg_style(an)[0]
            ax.set_title(f"{label_alg} - {pl} sweep",
                         pad=_TITLE_PAD, fontsize=8)
            ax.set_xlabel("Iteration", fontsize=7.5)
            ax.set_ylabel(r"relF", fontsize=7.5)

            for fac, relF_m, is_div in curves:
                t = (np.log10(float(fac)) - log_lo) / log_span
                color = cmap(0.05 + 0.90 * t)

                relF_m = np.asarray(relF_m, dtype=float)
                if is_div:
                    K = min(80, len(relF_m))
                    if K > 0:
                        ax.plot(range(1, K + 1),
                                np.clip(relF_m[:K], _FLOOR, None),
                                color=(0.72, 0.72, 0.72), ls="--",
                                lw=0.7, alpha=0.55)
                        ax.plot(K, max(relF_m[K - 1], _FLOOR),
                                "x", color="red", markersize=4,
                                markeredgewidth=1.0, zorder=5)
                else:
                    steps = np.arange(1, len(relF_m) + 1)
                    ax.plot(steps, np.clip(relF_m, _FLOOR, None),
                            color=color, lw=1.2, alpha=0.88)

        for si in range(n_sub, n_rows * n_cols):
            axes[si // n_cols, si % n_cols].set_visible(False)

        fig.suptitle(f"{group_label} - {_clean_title(obj_name)}",
                     fontsize=9, y=1.01)
        plt.tight_layout(rect=[0, 0, 0.90, 0.98])

        import matplotlib.colors as mcolors
        norm = mcolors.LogNorm(vmin=float(min(all_factors)),
                               vmax=float(max(all_factors)))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = fig.add_axes([0.92, 0.06, 0.02, 0.88])
        cbar = fig.colorbar(sm, cax=cax)
        mul_sym = r"$\times$default"
        cbar.set_label(r"factor ($\times$ default)", fontsize=7.5)
        cbar.ax.tick_params(labelsize=6.5)

        safe = group_label.replace(" ", "_").lower()
        base = os.path.join(fig_dir, f"{safe}_{obj_name}")
        _save(fig, base)
        plt.close(fig)


def fig_param_sweep_combined(all_entries_by_func: dict,
                              group_label: str,
                              func_names: list,
                              results_dir: str,
                              cmap_name: str = "viridis") -> None:
    """
    Combined parameter-sweep figure: rows=functions, cols=algorithms.
    One big figure per algorithm group, covering all functions.
    """
    fig_dir = os.path.join(results_dir, "paper", "param_sweep")
    os.makedirs(fig_dir, exist_ok=True)

    first_entries = None
    for fn in func_names:
        if fn in all_entries_by_func and all_entries_by_func[fn]:
            first_entries = all_entries_by_func[fn]
            break
    if first_entries is None:
        return

    n_alg_sub = len(first_entries)
    n_funcs = len(func_names)

    all_factors = first_entries[0]["factors"]
    cmap = plt.colormaps[cmap_name]
    log_lo = np.log10(float(min(all_factors)))
    log_hi = np.log10(float(max(all_factors)))
    log_span = max(log_hi - log_lo, 1e-6)

    with _rc():
        fig, axes = plt.subplots(
            n_funcs, n_alg_sub,
            figsize=(_COL2_W + 1.0, 2.0 * n_funcs + 0.6),
            squeeze=False)

        for row, obj_name in enumerate(func_names):
            entries = all_entries_by_func.get(obj_name, [])
            for col, entry in enumerate(entries):
                ax = axes[row, col]
                an = entry["alg_name"]
                pl = entry["param_label"]
                curves = entry["curves"]

                _format_ax(ax, "semilogy")
                label_alg = get_alg_style(an)[0]
                if row == 0:
                    ax.set_title(f"{label_alg} - {pl} sweep",
                                 pad=_TITLE_PAD, fontsize=7.5)
                if col == 0:
                    ax.set_ylabel(_clean_title(obj_name) + r"  relF",
                                  fontsize=7)
                else:
                    ax.set_ylabel("")
                if row == n_funcs - 1:
                    ax.set_xlabel("Iteration", fontsize=7)

                for fac, relF_m, is_div in curves:
                    t = (np.log10(float(fac)) - log_lo) / log_span
                    color = cmap(0.05 + 0.90 * t)
                    relF_m = np.asarray(relF_m, dtype=float)
                    if is_div:
                        K = min(80, len(relF_m))
                        if K > 0:
                            ax.plot(range(1, K + 1),
                                    np.clip(relF_m[:K], _FLOOR, None),
                                    color=(0.72, 0.72, 0.72), ls="--",
                                    lw=0.7, alpha=0.55)
                            ax.plot(K, max(relF_m[K - 1], _FLOOR),
                                    "x", color="red", markersize=4,
                                    markeredgewidth=1.0, zorder=5)
                    else:
                        steps = np.arange(1, len(relF_m) + 1)
                        ax.plot(steps, np.clip(relF_m, _FLOOR, None),
                                color=color, lw=1.2, alpha=0.88)

            for col in range(len(entries), n_alg_sub):
                axes[row, col].set_visible(False)

        title_map = {
            "disgrem_family": "DisGrem Family",
            "first-order": "First-Order Methods",
            "second-order": "Second-Order Methods",
        }
        fig.suptitle(title_map.get(group_label, group_label),
                     fontsize=10, y=1.01)
        plt.tight_layout(rect=[0, 0, 0.90, 0.98])

        import matplotlib.colors as mcolors
        norm = mcolors.LogNorm(vmin=float(min(all_factors)),
                               vmax=float(max(all_factors)))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = fig.add_axes([0.92, 0.06, 0.018, 0.88])
        cbar = fig.colorbar(sm, cax=cax)
        mul_sym = r"$\times$default"
        cbar.set_label(r"factor ($\times$ default)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        safe = group_label.replace(" ", "_").lower()
        base = os.path.join(fig_dir, f"{safe}_combined")
        _save(fig, base)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Communication: comm-cost-to-reach-precision profile (dimension gradient)
# ─────────────────────────────────────────────────────────────────────────────

def fig_comm_profile(profile_data: dict, obj_name: str,
                     dims: list, tol_levels: list,
                     results_dir: str,
                     cmap_name: str = "viridis") -> None:
    """
    Communication-cost profile: comm needed to reach each relF threshold.

    One figure per function, with one subplot per algorithm.
    Within each subplot, gradient-coloured curves for different dimensions.

    Parameters
    ----------
    profile_data : {alg_name: {dim: list_of_comm_values_per_tol}}
        comm_values[i] = comm cost (MB) to reach tol_levels[i], or NaN.
    obj_name   : function name (for title & filename)
    dims       : sorted list of dimensions [20, 30, 40, 50]
    tol_levels : sorted list of tolerances [1e-3, ..., 1e-10]
    results_dir: output root
    cmap_name  : sequential colormap for dimension gradient
    """
    fig_dir = os.path.join(results_dir, "paper", "comm_profile")
    os.makedirs(fig_dir, exist_ok=True)

    alg_names = list(profile_data.keys())
    n_sub = len(alg_names)
    if n_sub == 0:
        return
    n_cols = min(n_sub, 2)
    n_rows = int(np.ceil(n_sub / n_cols))

    cmap = plt.colormaps[cmap_name]
    n_dim = len(dims)

    with _rc():
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(_COL2_W, _CELL_H * n_rows + 0.6),
            squeeze=False)

        for si, an in enumerate(alg_names):
            ax = axes[si // n_cols, si % n_cols]
            label_alg = get_alg_style(an)[0]
            ax.set_title(label_alg, pad=_TITLE_PAD, fontsize=8)

            tol_exp = [-np.log10(t) for t in tol_levels]
            ax.set_xlabel(r"Precision $-\log_{10}(\mathrm{relF})$"
                          if _USE_USETEX else
                          "Precision -log10(relF)", fontsize=7.5)
            ax.set_ylabel("Comm. cost (MB)", fontsize=7.5)
            ax.grid(True, which="both", ls="--", alpha=_GRID_ALPHA)

            for di, dim in enumerate(dims):
                t = di / max(n_dim - 1, 1)
                color = cmap(0.20 + 0.65 * t)
                comm_vals = np.asarray(profile_data[an].get(dim, []),
                                       dtype=float)
                if len(comm_vals) == 0:
                    continue
                valid = np.isfinite(comm_vals)
                if not np.any(valid):
                    ax.plot([], [], color=color, marker="o", markersize=3,
                            lw=1.3, label=f"d={dim} (fail)")
                    continue

                x_plot = np.asarray(tol_exp)[valid]
                y_plot = comm_vals[valid]
                ax.plot(x_plot, y_plot, color=color, marker="o",
                        markersize=3.5, lw=1.3, alpha=0.88,
                        label=f"d={dim}")

                # Mark unreached thresholds with x at right edge
                for ui in np.where(~valid)[0]:
                    ax.plot(tol_exp[ui], ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 1.0,
                            "x", color=color, markersize=4,
                            markeredgewidth=1.0, alpha=0.6)

            ax.legend(fontsize=6.5, loc="upper left", frameon=True,
                      handlelength=1.4, labelspacing=0.3)

        for si in range(n_sub, n_rows * n_cols):
            axes[si // n_cols, si % n_cols].set_visible(False)

        fig.suptitle(f"Comm. Cost Profile - {_clean_title(obj_name)}",
                     fontsize=9, y=1.01)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        base = os.path.join(fig_dir, f"comm_profile_{obj_name}")
        _save(fig, base)
        plt.close(fig)


def fig_comm_summary_table(table_data: dict, dims: list,
                            tol_key_levels: list,
                            results_dir: str) -> None:
    """
    Compact summary table: comm cost at 3 key thresholds across algorithms.

    Parameters
    ----------
    table_data : {obj_name: {alg_name: {dim: {tol: float(MB)}}}}
    dims       : list of dimensions
    tol_key_levels : 3 representative tolerances [1e-3, 1e-6, 1e-9]
    results_dir: output root
    """
    fig_dir = os.path.join(results_dir, "paper", "comm_profile")
    os.makedirs(fig_dir, exist_ok=True)

    obj_names = list(table_data.keys())
    if not obj_names:
        return

    # One table per dimension
    for dim in dims:
        all_algs = list(table_data[obj_names[0]].keys())
        n_alg = len(all_algs)
        n_func = len(obj_names)
        n_tol = len(tol_key_levels)
        n_col = n_func * n_tol

        mat = np.full((n_alg, n_col), np.nan)
        for ai, an in enumerate(all_algs):
            ci = 0
            for on in obj_names:
                for tol in tol_key_levels:
                    val = table_data[on].get(an, {}).get(dim, {}).get(tol, np.nan)
                    mat[ai, ci] = val
                    ci += 1

        with _rc():
            fig_w = max(0.65 * n_col + 2.0, 6.0)
            fig_h = max(0.35 * n_alg + 1.5, 2.5)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            from matplotlib.colors import LogNorm
            valid_vals = mat[np.isfinite(mat)]
            if len(valid_vals) == 0:
                plt.close(fig)
                continue
            vmin = max(float(np.nanmin(valid_vals)), 1e-3)
            vmax = float(np.nanmax(valid_vals))
            im = ax.imshow(mat, cmap="YlOrRd", aspect="auto",
                           norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 2)))

            for ai in range(n_alg):
                for ci_idx in range(n_col):
                    val = mat[ai, ci_idx]
                    if np.isfinite(val):
                        txt = f"{val:.1f}" if val >= 1 else f"{val:.2f}"
                        tc = "white" if val > vmax * 0.5 else "black"
                        ax.text(ci_idx, ai, txt, ha="center", va="center",
                                fontsize=6, color=tc)
                    else:
                        ax.text(ci_idx, ai, "-", ha="center", va="center",
                                fontsize=6, color="gray")

            # Column labels: (function x tol) pairs
            col_labels = []
            for on in obj_names:
                for tol in tol_key_levels:
                    col_labels.append(
                        f"{_clean_title(on, short=True)}\n{tol:.0e}")

            row_labels = [get_alg_style(an)[0] for an in all_algs]

            ax.set_xticks(range(n_col))
            ax.set_xticklabels(col_labels, fontsize=5.5, rotation=0, ha="center")
            ax.set_yticks(range(n_alg))
            ax.set_yticklabels(row_labels, fontsize=7)
            ax.set_title(f"Comm. Cost (MB) - d={dim}", fontsize=9, pad=6)

            cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.04)
            cbar.set_label("MB", fontsize=7)
            cbar.ax.tick_params(labelsize=6)

            plt.tight_layout()
            base = os.path.join(fig_dir, f"comm_table_d{dim}")
            _save(fig, base)
            plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Communication: Ce benefit (full vs compressed) - Part 1
# ─────────────────────────────────────────────────────────────────────────────

def fig_ce_benefit(all_data: dict, ce_pairs: list,
                   tol_levels: list, results_dir: str) -> None:
    """
    Precision-vs-communication-cost plot showing Ce savings.

    2x2 grid (one subplot per function).
    Each subplot shows one curve per algorithm (full vs Ce variant).

    Parameters
    ----------
    all_data   : {obj_name: {alg_name: [comm_at_tol_1, ..., comm_at_tol_N]}}
    ce_pairs   : [(no_ce, with_ce), ...] algorithm pair tuples
    tol_levels : list of tolerance thresholds
    results_dir: output root
    """
    fig_dir = os.path.join(results_dir, "paper", "comm_profile")
    os.makedirs(fig_dir, exist_ok=True)

    obj_names = list(all_data.keys())
    if not obj_names:
        return

    n = len(obj_names)
    n_cols = min(n, 2)
    n_rows = int(np.ceil(n / n_cols))

    alg_list = []
    for no_ce, with_ce in ce_pairs:
        alg_list.extend([no_ce, with_ce])

    with _rc():
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(_COL2_W, _CELL_H * n_rows + 0.6),
            squeeze=False)

        handles_global, labels_global = [], []

        for idx, obj_name in enumerate(obj_names):
            ax = axes[idx // n_cols, idx % n_cols]
            ax.set_title(_clean_title(obj_name), pad=_TITLE_PAD, fontsize=8)
            tol_exp = [-np.log10(t) for t in tol_levels]
            ax.set_xlabel(r"Precision $-\log_{10}(\mathrm{relF})$"
                          if _USE_USETEX else
                          "Precision -log10(relF)", fontsize=7.5)
            ax.set_ylabel("Comm. cost (MB)", fontsize=7.5)
            ax.grid(True, which="both", ls="--", alpha=_GRID_ALPHA)

            for alg_name in alg_list:
                comm_vals = np.asarray(
                    all_data.get(obj_name, {}).get(alg_name, []),
                    dtype=float)
                if len(comm_vals) == 0:
                    continue

                label, color, ls = get_alg_style(alg_name)
                valid = np.isfinite(comm_vals)
                if not np.any(valid):
                    h, = ax.plot([], [], color=color, ls=ls, lw=1.5,
                                 label=f"{label} (fail)")
                    continue

                x_plot = np.asarray(tol_exp)[valid]
                y_plot = comm_vals[valid]
                is_ce = "Ce" in alg_name
                h, = ax.plot(x_plot, y_plot, color=color, ls=ls,
                             marker="s" if is_ce else "o",
                             markersize=4, lw=_LW_OURS if not is_ce else 1.5,
                             alpha=0.9, label=label)
                if label not in labels_global:
                    handles_global.append(h)
                    labels_global.append(label)

            saving_lines = []
            for no_ce, with_ce in ce_pairs:
                full_v = np.asarray(
                    all_data.get(obj_name, {}).get(no_ce, []), dtype=float)
                ce_v = np.asarray(
                    all_data.get(obj_name, {}).get(with_ce, []), dtype=float)
                K = min(len(full_v), len(ce_v))
                if K == 0:
                    continue
                valid = np.isfinite(full_v[:K]) & np.isfinite(ce_v[:K]) & (full_v[:K] > 0)
                if not np.any(valid):
                    continue
                avg_saving = 100.0 * np.nanmean(1.0 - ce_v[:K][valid] / full_v[:K][valid])
                _, _, label_ce = no_ce.partition("Ada")
                tag = "CeAda" if label_ce else "Ce"
                saving_lines.append(f"{tag}: {avg_saving:.0f}%")
            if saving_lines:
                ax.text(0.03, 0.96, "Avg. saving " + ", ".join(saving_lines),
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=6.5,
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white", edgecolor="0.8",
                                  alpha=0.85))

        for idx in range(n, n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)

        if handles_global:
            fig.legend(handles_global, labels_global,
                       loc="lower center",
                       ncol=min(len(handles_global), 4),
                       bbox_to_anchor=(0.5, 0.0),
                       frameon=True,
                       handlelength=_LEG_HLEN,
                       columnspacing=_LEG_COLSPC,
                       fontsize=_LEG_FONT)

        plt.tight_layout(rect=[0, 0.07, 1, 1.0])
        base = os.path.join(fig_dir, "ce_benefit")
        _save(fig, base)
        plt.close(fig)

    # ── Savings table ────────────────────────────────────────────────────
    tbl_dir = os.path.join(results_dir, "data_log")
    os.makedirs(tbl_dir, exist_ok=True)
    rows = [["Function", "Full Alg", "Ce Alg", "Precision"] +
            ["CommFull(MB)", "CommCe(MB)", "Saving(%)"]]
    for obj_name in obj_names:
        for no_ce, with_ce in ce_pairs:
            full_v = np.asarray(
                all_data.get(obj_name, {}).get(no_ce, []), dtype=float)
            ce_v = np.asarray(
                all_data.get(obj_name, {}).get(with_ce, []), dtype=float)
            for ti, tol in enumerate(tol_levels):
                fv = full_v[ti] if ti < len(full_v) else np.nan
                cv = ce_v[ti] if ti < len(ce_v) else np.nan
                if np.isfinite(fv) and np.isfinite(cv) and fv > 0:
                    pct = 100.0 * (1.0 - cv / fv)
                    rows.append([obj_name, no_ce, with_ce, f"{tol:.0e}",
                                 f"{fv:.2f}", f"{cv:.2f}", f"{pct:.1f}"])
                else:
                    rows.append([obj_name, no_ce, with_ce, f"{tol:.0e}",
                                 f"{fv:.2f}" if np.isfinite(fv) else "NaN",
                                 f"{cv:.2f}" if np.isfinite(cv) else "NaN",
                                "-"])

    import csv
    csv_path = os.path.join(tbl_dir, "ce_savings_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    print(f"  [Saved] {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Communication: ablation (Klazy sweep / compression sweep) - Part 2
# ─────────────────────────────────────────────────────────────────────────────

def fig_comm_ablation(curves_by_func: dict, obj_names: list,
                      suptitle: str, filename: str,
                      results_dir: str,
                      cmap_name: str = "Blues") -> None:
    """
    relF vs commCost gradient curves for ablation studies.

    1x4 (or 2x2 if >4 functions) layout.

    Parameters
    ----------
    curves_by_func : {obj_name: [(label, relF_arr, commCost_arr), ...]}
        Each entry is a config/param value.  relF_arr/commCost_arr may be
        None (skipped).
    obj_names : list of function names for ordering
    suptitle  : figure super-title
    filename  : output filename (without extension)
    results_dir : output root
    cmap_name   : colormap for gradient colouring
    """
    fig_dir = os.path.join(results_dir, "paper", "comm_profile")
    os.makedirs(fig_dir, exist_ok=True)

    n = len(obj_names)
    n_cols = min(n, 2)
    n_rows = int(np.ceil(n / n_cols))

    cmap = plt.colormaps[cmap_name]

    with _rc():
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(_COL2_W, _CELL_H * n_rows + 0.6),
            squeeze=False)

        handles_global, labels_global = [], []

        for idx, obj_name in enumerate(obj_names):
            if obj_name not in curves_by_func:
                continue
            curves = curves_by_func[obj_name]
            ax = axes[idx // n_cols, idx % n_cols]
            _format_ax(ax, "semilogy")
            ax.set_title(_clean_title(obj_name), pad=_TITLE_PAD, fontsize=8)
            ax.set_xlabel("Comm. cost (MB)", fontsize=7.5)
            ax.set_ylabel("relF", fontsize=7.5)

            n_curves = len(curves)
            for ci, (label, relF, comm) in enumerate(curves):
                if relF is None or comm is None:
                    continue
                t = ci / max(n_curves - 1, 1)
                color = cmap(0.15 + 0.80 * t)

                K = min(len(relF), len(comm))
                relF_c = np.clip(_running_min(relF[:K]), _FLOOR, None)
                comm_c = comm[:K]
                valid = np.isfinite(relF_c) & np.isfinite(comm_c) & (relF_c > 0)
                if not np.any(valid):
                    h, = ax.plot([], [], color=color, lw=1.3, label=label)
                else:
                    x_valid = comm_c[valid]
                    y_valid = relF_c[valid]
                    order = np.argsort(x_valid)
                    x_valid = x_valid[order]
                    y_valid = _running_min(y_valid[order])
                    h, = ax.plot(x_valid, y_valid,
                                 color=color, lw=1.4, alpha=0.88, label=label)
                if label not in labels_global:
                    handles_global.append(h)
                    labels_global.append(label)

        for idx in range(n, n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)

        if handles_global:
            fig.legend(handles_global, labels_global,
                       loc="lower center",
                       ncol=min(len(handles_global), 5),
                       bbox_to_anchor=(0.5, 0.0),
                       frameon=True,
                       handlelength=_LEG_HLEN,
                       columnspacing=_LEG_COLSPC,
                       fontsize=_LEG_FONT)

        plt.tight_layout(rect=[0, 0.08, 1, 1.0])
        base = os.path.join(fig_dir, filename)
        _save(fig, base)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Adaptive mechanism: M trajectory, Ada vs Fixed-M, initial-M robustness
# ─────────────────────────────────────────────────────────────────────────────

def fig_ada_m_trajectory(logs: dict, obj_names: list,
                          results_dir: str) -> None:
    """
    M-trajectory plot: 2x2 grid (one subplot per function).

    Each subplot shows:
      - AdaDisGrem  M(t) trajectory  (solid blue)
      - CeAdaDisGrem M(t) trajectory (dashed teal)
      - DisGrem fixed M              (horizontal dash-dot grey)

    Parameters
    ----------
    logs : {obj_name: {alg_name: out_dict}}
        Must contain at least AdaDisGrem; DisGrem and CeAdaDisGrem optional.
    obj_names : list of function names (should be length 4)
    results_dir : output root
    """
    fig_dir = os.path.join(results_dir, "paper", "ada_mechanism")
    os.makedirs(fig_dir, exist_ok=True)

    n = len(obj_names)
    n_cols = min(n, 2)
    n_rows = int(np.ceil(n / n_cols))

    with _rc():
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(_COL2_W, _CELL_H * n_rows + 0.5),
            squeeze=False)

        for idx, obj_name in enumerate(obj_names):
            ax = axes[idx // n_cols, idx % n_cols]
            ax.set_title(_clean_title(obj_name), pad=_TITLE_PAD, fontsize=8)
            ax.set_xlabel("Iteration", fontsize=7.5)
            ax.set_ylabel(r"$M$", fontsize=7.5)
            ax.grid(True, which="both", ls="--", alpha=_GRID_ALPHA)

            obj_log = logs.get(obj_name, {})

            # DisGrem fixed M (horizontal line)
            dg = obj_log.get("DisGrem", {})
            m_dg = np.asarray(dg.get("Mavg", []), dtype=float)
            if len(m_dg) > 0:
                m_fixed = float(m_dg[0])
                ax.axhline(m_fixed, color=(0.5, 0.5, 0.5), ls="-.",
                           lw=1.2, alpha=0.7, zorder=2,
                           label=f"DisGrem (M={m_fixed:.1f})")

            # AdaDisGrem trajectory
            ada = obj_log.get("AdaDisGrem", {})
            m_ada = np.asarray(ada.get("Mavg", []), dtype=float)
            if len(m_ada) > 0:
                valid = np.isfinite(m_ada)
                steps = np.arange(1, len(m_ada) + 1)
                label_ada, col_ada, _ = get_alg_style("AdaDisGrem")
                ax.plot(steps[valid], m_ada[valid], color=col_ada,
                        ls="-", lw=1.8, alpha=0.9, zorder=3,
                        label=label_ada)

            # CeAdaDisGrem trajectory
            ceada = obj_log.get("CeAdaDisGrem", {})
            m_ceada = np.asarray(ceada.get("Mavg", []), dtype=float)
            if len(m_ceada) > 0:
                valid = np.isfinite(m_ceada)
                steps = np.arange(1, len(m_ceada) + 1)
                label_ce, col_ce, _ = get_alg_style("CeAdaDisGrem")
                ax.plot(steps[valid], m_ceada[valid], color=col_ce,
                        ls="--", lw=1.6, alpha=0.85, zorder=3,
                        label=label_ce)

            ax.legend(fontsize=6, loc="best", frameon=True,
                      handlelength=1.8, labelspacing=0.3)

        for idx in range(n, n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)

        plt.tight_layout()
        base = os.path.join(fig_dir, "ada_m_trajectory")
        _save(fig, base)
        plt.close(fig)


def fig_ada_vs_fixed_m(ada_log: dict, fixed_m_logs: dict,
                        m_factors: list, obj_names: list,
                        results_dir: str) -> None:
    """
    Ada vs Fixed-M convergence comparison: 2x2 grid.

    Each subplot (one function):
      - Grey gradient curves: DisGrem at each fixed M factor
      - Bold blue curve: AdaDisGrem (auto-tuned M)

    Parameters
    ----------
    ada_log     : {obj_name: out_dict}  AdaDisGrem results
    fixed_m_logs: {obj_name: {m_factor: out_dict}} DisGrem at various fixed M
    m_factors   : sorted list of M multiplier factors
    obj_names   : list of 4 function names
    results_dir : output root
    """
    fig_dir = os.path.join(results_dir, "paper", "ada_mechanism")
    os.makedirs(fig_dir, exist_ok=True)

    n = len(obj_names)
    n_cols = min(n, 2)
    n_rows = int(np.ceil(n / n_cols))
    cmap = plt.colormaps["Greys"]

    with _rc():
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(_COL2_W, _CELL_H * n_rows + 0.5),
            squeeze=False)

        for idx, obj_name in enumerate(obj_names):
            ax = axes[idx // n_cols, idx % n_cols]
            _format_ax(ax, "semilogy")
            ax.set_title(_clean_title(obj_name), pad=_TITLE_PAD, fontsize=8)
            ax.set_xlabel("Iteration", fontsize=7.5)
            ax.set_ylabel(r"relF", fontsize=7.5)

            # Fixed-M curves (grey gradient)
            n_f = len(m_factors)
            for fi, mf in enumerate(m_factors):
                t = (fi + 1) / (n_f + 1)
                color = cmap(0.30 + 0.55 * t)
                out = fixed_m_logs.get(obj_name, {}).get(mf, {})
                relF = np.asarray(out.get("relF", []), dtype=float)
                if len(relF) == 0:
                    continue
                steps = np.arange(1, len(relF) + 1)
                relF_c = np.clip(relF, _FLOOR, None)
                ax.plot(steps, relF_c, color=color, lw=0.9,
                        alpha=0.7, zorder=2,
                        label=f"M={mf:.2g}" + r"$\times$")

            # AdaDisGrem curve (bold)
            ada_out = ada_log.get(obj_name, {})
            relF_ada = np.asarray(ada_out.get("relF", []), dtype=float)
            if len(relF_ada) > 0:
                steps = np.arange(1, len(relF_ada) + 1)
                label_ada, col_ada, _ = get_alg_style("AdaDisGrem")
                ax.plot(steps, np.clip(relF_ada, _FLOOR, None),
                        color=col_ada, lw=2.2, alpha=0.92, zorder=4,
                        label=label_ada)

            ax.legend(fontsize=5.5, loc="upper right", frameon=True,
                      ncol=2, handlelength=1.4, labelspacing=0.25,
                      columnspacing=0.8)

        for idx in range(n, n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)

        plt.tight_layout()
        base = os.path.join(fig_dir, "ada_vs_fixed_m")
        _save(fig, base)
        plt.close(fig)


def fig_ada_init_m_robust(init_m_logs: dict, init_m_factors: list,
                           obj_names: list,
                           results_dir: str) -> None:
    """
    Initial-M robustness: M trajectory from different starting M values.

    2x2 grid; each subplot shows multiple M(t) curves for AdaDisGrem
    launched with different initial M, demonstrating convergence to
    a similar operating point regardless of initialisation.

    Parameters
    ----------
    init_m_logs    : {obj_name: {init_factor: out_dict}}
    init_m_factors : sorted list of initial-M multiplier factors
    obj_names      : 4 function names
    results_dir    : output root
    """
    fig_dir = os.path.join(results_dir, "paper", "ada_mechanism")
    os.makedirs(fig_dir, exist_ok=True)

    n = len(obj_names)
    n_cols = min(n, 2)
    n_rows = int(np.ceil(n / n_cols))
    cmap = plt.colormaps["Blues"]

    with _rc():
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(_COL2_W, _CELL_H * n_rows + 0.5),
            squeeze=False)

        n_f = len(init_m_factors)
        for idx, obj_name in enumerate(obj_names):
            ax = axes[idx // n_cols, idx % n_cols]
            ax.set_title(_clean_title(obj_name), pad=_TITLE_PAD, fontsize=8)
            ax.set_xlabel("Iteration", fontsize=7.5)
            ax.set_ylabel(r"$M$", fontsize=7.5)
            ax.grid(True, which="both", ls="--", alpha=_GRID_ALPHA)

            for fi, mf in enumerate(init_m_factors):
                t = (fi + 1) / (n_f + 1)
                color = cmap(0.25 + 0.60 * t)
                out = init_m_logs.get(obj_name, {}).get(mf, {})
                m_arr = np.asarray(out.get("Mavg", []), dtype=float)
                if len(m_arr) == 0:
                    continue
                steps = np.arange(1, len(m_arr) + 1)
                valid = np.isfinite(m_arr)
                mul_sym = r"$\times$"
                ax.plot(steps[valid], m_arr[valid], color=color,
                        lw=1.3, alpha=0.85,
                        label=f"init M={mf:.2g}{mul_sym}")

            ax.legend(fontsize=5.5, loc="best", frameon=True,
                      handlelength=1.4, labelspacing=0.3)

        for idx in range(n, n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)

        plt.tight_layout()
        base = os.path.join(fig_dir, "ada_init_m_robust")
        _save(fig, base)
        plt.close(fig)
