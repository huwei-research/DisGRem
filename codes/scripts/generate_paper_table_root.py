"""
generate_paper_table.py - Generate LaTeX-formatted summary table from results/summary.txt

Usage:  python generate_paper_table.py
Output: results/paper_table.tex   (main comparison table)
        results/paper_table_2nd.tex (second-order comparison table)

Table structure
---------------
Main table (Table I):
  Rows = algorithms, grouped as: Proposed (2nd-order) | Proposed (quasi) | 1st-order baselines
  Cols = objectives x {Steps, relF}

Second-order table (Table II):
  Rows = algorithms, grouped as: Proposed (2nd-order) | 2nd-order baselines
  Cols = objectives x {Steps, relF}
"""

import os
import re
import sys

_root        = os.path.dirname(os.path.abspath(__file__))
summary_path = os.path.join(_root, "results", "summary.txt")
out_main     = os.path.join(_root, "results", "paper_table.tex")
out_second   = os.path.join(_root, "results", "paper_table_2nd.tex")

# ── Parse summary.txt ──────────────────────────────────────────────────────
results = {}   # {obj_name: {alg_name: {steps, combo, relF, relX, comm}}}
current_obj  = None
header_seen  = False

if os.path.isfile(summary_path):
    with open(summary_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip()
            m = re.match(r"^Objective:\s+(\S+)", line)
            if m:
                current_obj = m.group(1)
                results[current_obj] = {}
                header_seen = False
                continue
            if current_obj and "Algorithm" in line and "Steps" in line:
                header_seen = True
                continue
            if current_obj and header_seen and "|" in line and not line.startswith("-"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 10:
                    try:
                        results[current_obj][parts[0]] = {
                            "steps": parts[2], "combo": parts[3],
                            "stdcombo": parts[4], "converged": parts[5],
                            "finalrelF": parts[6],
                            "relF":  parts[7], "relX":  parts[8], "comm": parts[9],
                        }
                    except Exception:
                        pass
                elif len(parts) >= 7:
                    try:
                        results[current_obj][parts[0]] = {
                            "steps": parts[2], "combo": parts[3],
                            "relF":  parts[4], "relX":  parts[5], "comm": parts[6],
                        }
                    except Exception:
                        pass
else:
    print(f"[Warning] {summary_path} not found - generating empty table skeleton.")

# ── Objective and algorithm configuration ─────────────────────────────────

# ── Main benchmark (9 functions) ──────────────────────────────────────────
# Convex (6):    ridge, quadbad, logsumexp, huber, linlog, logreg_real
# Non-convex (3): rosenbrock, styblinski_tang, logreg_ncvr
OBJ_MAIN = ["ridge", "quadbad",
            "logsumexp", "huber", "linlog", "logreg_real",
            "rosenbrock", "styblinski_tang", "logreg_ncvr"]

OBJ_LABEL = {
    "ridge":              r"\texttt{Ridge}",
    "quadbad":            r"\texttt{QuadBad}",
    "logsumexp":          r"\texttt{LSE}",
    "huber":              r"\texttt{Huber}",
    "linlog":             r"\texttt{LinLog}",
    "logreg_real":        r"\texttt{LR}",
    "rosenbrock":         r"\texttt{Rosen}",
    "styblinski_tang":    r"\texttt{ST}",
    "logreg_ncvr":        r"\texttt{LR-NC}",
}

# All algorithms appearing in the tables
ALG_SHOW_MAIN   = ["DisGrem", "CeDisGrem", "AdaDisGrem", "CeAdaDisGrem",
                   "DisGreQm",
                   "EXTRA", "DIGing",
                   "DQM", "ESOM", "SONATA", "NetworkGIANT"]

ALG_SHOW_SECOND = ["DisGrem", "CeDisGrem", "AdaDisGrem", "CeAdaDisGrem",
                   "DQM", "ESOM", "SONATA", "NetworkGIANT"]

ALG_LABEL = {
    "DisGrem":        r"\textbf{DisGrem}",
    "CeDisGrem":      r"\textbf{CeDisGrem}",
    "AdaDisGrem":     r"\textbf{AdaDisGrem}",
    "CeAdaDisGrem":   r"\textbf{CeAdaDisGrem}",
    "DisGreQm":       r"\textbf{DisGre$\mathbb{Q}$m}",
    "EXTRA":          "EXTRA",
    "DIGing":         "DIGing",
    "DQM":            "DQM",
    "ESOM":           "ESOM",
    "SONATA":         "SONATA",
    "NetworkGIANT":   "Net-GIANT",
    "DisQN":          "DisQN",
}

# Table group structure: (member algorithms, section label, rule before)
GROUPS_MAIN = [
    (["DisGrem", "CeDisGrem", "AdaDisGrem", "CeAdaDisGrem"],
     "Proposed (2nd-order)", True),
    (["DisGreQm"],
     "Proposed (quasi-Newton)", False),
    (["EXTRA", "DIGing"],
     "First-order baselines", True),
    (["DQM", "ESOM", "SONATA", "NetworkGIANT"],
     "Second-order baselines", True),
]

GROUPS_SECOND = [
    (["DisGrem", "CeDisGrem", "AdaDisGrem", "CeAdaDisGrem"],
     "Proposed (2nd-order)", True),
    (["DQM", "ESOM", "SONATA", "NetworkGIANT"],
     "Second-order baselines", True),
]


# ── Formatting helpers ─────────────────────────────────────────────────────
def fmt_sci(s: str) -> str:
    """Format a floating-point string as compact LaTeX scientific notation."""
    s = s.strip()
    try:
        v = float(s)
        if not (v == v):          # NaN
            return r"---"
        if v < 1e-14:
            return r"$<10^{-14}$"
        if v < 1e-12:
            return r"$<10^{-12}$"
        e = int(f"{v:.2e}".split("e")[1])
        m = v / 10 ** e
        if abs(m - 1.0) < 0.12:
            return fr"$10^{{{e}}}$"
        return f"${m:.1f}" + r"{\times}10^{" + f"{e}" + "}$"
    except Exception:
        return s if s not in ("", "--", "nan") else "---"


def fmt_steps(s: str) -> str:
    s = s.strip()
    try:
        return str(int(float(s)))
    except Exception:
        return "---"


def _best_in_col(data: dict, obj: str, alg_list: list, field: str) -> float:
    """Return the minimum numeric value for (obj, field) across alg_list."""
    vals = []
    for alg in alg_list:
        raw = data.get(obj, {}).get(alg, {}).get(field, "nan")
        try:
            vals.append(float(raw))
        except Exception:
            pass
    return min(vals) if vals else float("nan")


def fmt_comm(s: str) -> str:
    """Format communication cost (MB) for table display."""
    s = s.strip()
    try:
        v = float(s)
        if not (v == v):
            return "---"
        if v < 0.01:
            return f"${v:.2e}$".replace("e-0", r"{\times}10^{-").replace("e+0", r"{\times}10^{") + "}$" if "e" in f"{v:.2e}" else f"{v:.3f}"
        if v < 100:
            return f"{v:.2f}"
        return f"{v:.1f}"
    except Exception:
        return "---"


def _global_best_in_col(data: dict, obj: str, all_algs: list, field: str) -> float:
    """Return the minimum numeric value for (obj, field) across ALL algorithms."""
    vals = []
    for alg in all_algs:
        raw = data.get(obj, {}).get(alg, {}).get(field, "nan")
        try:
            vals.append(float(raw))
        except Exception:
            pass
    return min(vals) if vals else float("nan")


def _build_table(obj_list, alg_list, groups, caption, label, caption_note=""):
    """Build a complete LaTeX table string."""
    obj_avail = [o for o in obj_list if o in results]
    n_obj = len(obj_avail)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    cap = (f"\\caption{{{caption} "
           r"Steps: iterations to convergence (combo$<10^{-12}$) or max budget. "
           r"relF: minimum relative function-value error. "
           r"Comm: average communication cost (MB). "
           r"\textbf{Bold}: best value across all algorithms."
           + (f" {caption_note}" if caption_note else "")
           + r"}}")
    lines.append(cap)
    lines.append(fr"\label{{{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{")

    cols = "l" + "rrr" * n_obj
    lines.append(fr"\begin{{tabular}}{{{cols}}}")
    lines.append(r"\toprule")

    # Header row 1 - objective names spanning 3 cols each
    hdr1 = "Algorithm"
    for o in obj_avail:
        hdr1 += r" & \multicolumn{3}{c}{" + OBJ_LABEL.get(o, o) + "}"
    lines.append(hdr1 + r" \\")
    lines.append(r"\cmidrule(lr){2-" + str(1 + 3 * n_obj) + r"}")

    # Header row 2 - Steps | relF | Comm per objective
    hdr2 = ""
    for _ in obj_avail:
        hdr2 += r" & \footnotesize Steps & \footnotesize relF & \footnotesize Comm"
    lines.append("Algorithm" + hdr2 + r" \\")
    lines.append(r"\midrule")

    # Data rows
    first_group = True
    for grp_algs, grp_label, add_rule in groups:
        if not first_group and add_rule:
            lines.append(r"\midrule")
        first_group = False
        lines.append(
            fr"\multicolumn{{{1 + 3*n_obj}}}{{l}}{{\textit{{{grp_label}}}}} \\[-2pt]"
        )

        for alg in grp_algs:
            if alg not in alg_list:
                continue
            row = ALG_LABEL.get(alg, alg)
            for o in obj_avail:
                d = results.get(o, {}).get(alg, {})
                s_str = fmt_steps(d.get("steps", "---"))
                r_str = fmt_sci(d.get("relF",  "---"))
                c_str = fmt_comm(d.get("comm",  "---"))

                # Bold the global best steps across ALL algorithms
                best_steps = _global_best_in_col(results, o, alg_list, "steps")
                try:
                    if abs(float(d.get("steps", "nan")) - best_steps) < 1:
                        s_str = r"\textbf{" + s_str + "}"
                except Exception:
                    pass

                # Bold the global best relF
                best_relF = _global_best_in_col(results, o, alg_list, "relF")
                try:
                    v_relF = float(d.get("relF", "nan"))
                    if abs(v_relF - best_relF) / max(abs(best_relF), 1e-30) < 0.05:
                        r_str = r"\textbf{" + r_str + "}"
                except Exception:
                    pass

                # Bold the global best comm
                best_comm = _global_best_in_col(results, o, alg_list, "comm")
                try:
                    v_comm = float(d.get("comm", "nan"))
                    if abs(v_comm - best_comm) / max(abs(best_comm), 1e-30) < 0.05:
                        c_str = r"\textbf{" + c_str + "}"
                except Exception:
                    pass

                row += f" & {s_str} & {r_str} & {c_str}"
            lines.append(row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ── Generate tables ────────────────────────────────────────────────────────
os.makedirs(os.path.join(_root, "results", "main"), exist_ok=True)

table_main = _build_table(
    OBJ_MAIN, ALG_SHOW_MAIN, GROUPS_MAIN,
    caption=r"Benchmark results (proposed vs.\ first-order baselines).",
    label="tab:main_results",
)

table_second = _build_table(
    OBJ_MAIN, ALG_SHOW_SECOND, GROUPS_SECOND,
    caption="Second-order comparison.",
    label="tab:second_results",
)

with open(out_main, "w", encoding="utf-8") as fh:
    fh.write(table_main)
print(f"[Saved] {out_main}")

with open(out_second, "w", encoding="utf-8") as fh:
    fh.write(table_second)
print(f"[Saved] {out_second}")

print(f"  Objectives used : {[o for o in OBJ_MAIN if o in results]}")
print(f"  Algorithms (main): {ALG_SHOW_MAIN}")
print(f"  Algorithms (2nd) : {ALG_SHOW_SECOND}")
