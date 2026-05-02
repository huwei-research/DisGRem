"""
generate_paper_table.py - Generate comprehensive LaTeX summary table.

Reads all summary_*.txt files from results/ directory and produces a
publication-ready LaTeX table with columns:
  Algorithm | Steps | Comm (MB) | Time (s) | relF | Converged?

The table groups rows by objective function and bolds the best value
in each column within each group.

Outputs:
  results/paper_table.tex
"""

from __future__ import annotations
import os
import sys
import re
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)


_OBJ_LABEL = {
    "ridge":           r"\texttt{Ridge}",
    "quadbad":         r"\texttt{QuadBad}",
    "logsumexp":       r"\texttt{LSE}",
    "huber":           r"\texttt{Huber}",
    "linlog":          r"\texttt{LinLog}",
    "logreg_real":     r"\texttt{LR-real}",
    "rosenbrock":      r"\texttt{Rosen.}",
    "styblinski_tang": r"\texttt{Stybl.}",
    "logreg_ncvr":     r"\texttt{LR-NCVR}",
}

_OBJ_ORDER = [
    "ridge", "quadbad", "logsumexp", "huber", "linlog", "logreg_real",
    "rosenbrock", "styblinski_tang", "logreg_ncvr",
]


def _parse_summary(fpath: str) -> dict:
    """
    Parse a summary_*.txt file.

    Returns {alg_name: {"time": float, "steps": int, "combo": float,
                        "relF": float, "relX": float, "comm": float}}
    """
    data = {}
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "|" not in line or line.startswith("-") or line.startswith("Alg"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 7:
                continue
            alg_name = parts[0]
            if alg_name in ("Algorithm", ""):
                continue
            try:
                time_val = float(parts[1])
                steps = int(parts[2])
                combo = float(parts[3])
                relF = float(parts[4])
                relX = float(parts[5])
                comm = float(parts[6])
            except (ValueError, IndexError):
                continue
            data[alg_name] = {
                "time": time_val, "steps": steps,
                "combo": combo, "relF": relF, "relX": relX, "comm": comm,
            }
    return data


def generate_paper_table(results_dir: str = None) -> str:
    """
    Generate a complete LaTeX table from experiment results.

    Parameters
    ----------
    results_dir : path to results directory (default: <root>/results)

    Returns
    -------
    LaTeX table string
    """
    if results_dir is None:
        results_dir = os.path.join(_root, "results", "main")

    all_data = {}   # {obj_name: {alg_name: {...}}}
    for obj_name in _OBJ_ORDER:
        fpath = os.path.join(results_dir, f"summary_{obj_name}.txt")
        if os.path.isfile(fpath):
            data = _parse_summary(fpath)
            if data:
                all_data[obj_name] = data

    if not all_data:
        print("[generate_paper_table] No summary files found.")
        return ""

    # Collect all algorithm names (preserve order from first file)
    alg_names = []
    for obj_name in _OBJ_ORDER:
        if obj_name in all_data:
            for an in all_data[obj_name]:
                if an not in alg_names:
                    alg_names.append(an)

    # ── Build LaTeX ─────────────────────────────────────────────────────
    lines = []
    lines.append(r"\begin{table*}[tb]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comprehensive comparison across 9 benchmark functions.}")
    lines.append(r"\label{tab:full_comparison}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{l l r r r r c}")
    lines.append(r"\toprule")
    lines.append(r"Function & Algorithm & Steps & Comm (MB) & Time (s) & relF & Conv.\ \\")
    lines.append(r"\midrule")

    for obj_name in _OBJ_ORDER:
        if obj_name not in all_data:
            continue
        obj_label = _OBJ_LABEL.get(obj_name, r"\texttt{" + obj_name + "}")
        obj_data = all_data[obj_name]

        # Find best values for bolding
        metrics = ["steps", "comm", "time", "relF"]
        best = {}
        for m in metrics:
            vals = []
            for an in alg_names:
                if an in obj_data:
                    v = obj_data[an].get(m, np.nan)
                    if np.isfinite(v) and v > 0:
                        vals.append((v, an))
            if vals:
                best[m] = min(vals, key=lambda t: t[0])[1]

        first_row = True
        for an in alg_names:
            if an not in obj_data:
                continue
            d = obj_data[an]
            converged = d["relF"] < 1e-3

            def _fmt(key, fmt_str, is_int=False):
                v = d.get(key, np.nan)
                if not np.isfinite(v):
                    return r"\textemdash"
                s = f"{int(v)}" if is_int else fmt_str.format(v)
                if best.get(key) == an:
                    return r"\textbf{" + s + "}"
                return s

            obj_col = obj_label if first_row else ""
            conv_mark = r"\checkmark" if converged else r"$\times$"

            row = (f"  {obj_col} & {an} & "
                   f"{_fmt('steps', '', True)} & "
                   f"{_fmt('comm', '{:.2f}')} & "
                   f"{_fmt('time', '{:.2f}')} & "
                   f"{_fmt('relF', '{:.1e}')} & "
                   f"{conv_mark} \\\\")
            lines.append(row)
            first_row = False

        lines.append(r"\midrule")

    # Remove last \midrule and replace with \bottomrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    tex_content = "\n".join(lines) + "\n"

    # Save
    out_path = os.path.join(results_dir, "paper_table.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print(f"[Saved] {out_path}")

    return tex_content


if __name__ == "__main__":
    rd = sys.argv[1] if len(sys.argv) > 1 else None
    generate_paper_table(rd)
