"""
alg_bank.py - Algorithm registry.

Algorithm taxonomy
------------------
Second-order (exact Hessian):
  DisGrem, CeDisGrem, AdaDisGrem, CeAdaDisGrem   - proposed family
  DQM, ESOM                                        - 2nd-order baselines (classic)
  SONATA, NetworkGIANT                             - 2nd-order baselines (recent)

Quasi-Newton (gradient-only curvature approx):
  DisGreQm                                         - proposed quasi-Newton
  DisQN                                            - quasi-Newton baseline

First-order (gradient-only):
  EXTRA, DIGing                                    - 1st-order baselines

Mode groups
-----------
PRIMARY COMPARISON
  MainComp     - Our 4 + EXTRA + DIGing + DQM + ESOM + SONATA + NetworkGIANT
  MainCompCore - Our 4 + EXTRA + DIGing + DQM + ESOM
  CEStudy      - DisGrem + CeDisGrem + AdaDisGrem + CeAdaDisGrem

FULL / SUPPLEMENTARY
  PaperFull    - All proposed (5) + all baselines
  PaperSecond  - Our 4 + DQM + ESOM + SONATA + NetworkGIANT
  All          - every registered algorithm

THEMATIC SUBSETS
  DisGremFamily- all 5 proposed algorithms
  SecondOrder  - DQM + ESOM + SONATA + NetworkGIANT
  FirstOrder   - EXTRA + DIGing
  CeGroup      - CeDisGrem + CeAdaDisGrem
  AdaGroup     - AdaDisGrem + CeAdaDisGrem
"""

from __future__ import annotations
from solvers.disgrem.ce_disgrem import ce_disgrem
from solvers.disgrem.ce_ada_disgrem import ce_ada_disgrem
from solvers.disgrem.dis_greqm import dis_greqm
from solvers.baselines.extra import extra
from solvers.baselines.diging import diging
from solvers.baselines.dqm import dqm
from solvers.baselines.esom import esom
from solvers.baselines.dis_qn import dis_qn
from solvers.baselines.sonata import sonata
from solvers.baselines.network_giant import network_giant


def disgrem(x0, prm):
    """Formal DisGrem wrapper using the log-schedule CE-capable solver."""
    prm = dict(prm)
    prm["compressH"] = False
    prm["compressParam"] = None
    prm["compressor"] = "vector"
    prm["Klazy"] = 1
    prm["adaptive_ce"] = False
    prm["skip_H_premix"] = False
    prm["NC_mat_cap"] = 3
    prm["algName"] = "DisGrem"
    return ce_disgrem(x0, prm)


def ada_disgrem(x0, prm):
    """Formal AdaDisGrem wrapper using the log-schedule CE-capable solver."""
    prm = dict(prm)
    prm["compressH"] = False
    prm["compressParam"] = None
    prm["compressor"] = "vector"
    prm["Klazy"] = 1
    prm["adaptive_ce"] = False
    prm["skip_H_premix"] = False
    prm["NC_mat_cap"] = 3
    prm["algName"] = "AdaDisGrem"
    return ce_ada_disgrem(x0, prm)


def ce_disgrem_adaptive(x0, prm):
    """Communication-efficient DisGrem with adaptive CE parameters."""
    prm = dict(prm)
    prm["adaptive_ce"] = True
    prm["skip_H_premix"] = True
    prm["NC_mat_cap"] = 2
    prm["algName"] = "CeDisGrem"
    return ce_disgrem(x0, prm)


def ce_ada_disgrem_adaptive(x0, prm):
    """Communication-efficient AdaDisGrem with adaptive CE parameters."""
    prm = dict(prm)
    prm["adaptive_ce"] = True
    prm["skip_H_premix"] = True
    prm["NC_mat_cap"] = 2
    prm["algName"] = "CeAdaDisGrem"
    return ce_ada_disgrem(x0, prm)

# ── Master registry ────────────────────────────────────────────────────────
#   Each entry: (name, callable, color_rgb, latex_label)
#   Index:        0         1          2             3
_REGISTRY = [
    # ── Proposed second-order (BLUE gradient, light → dark) ───────────────
    ("DisGrem",        disgrem,          (0.65, 0.81, 0.94), r"DisGrem"),         # 0
    ("CeDisGrem",      ce_disgrem_adaptive, (0.39, 0.63, 0.85), r"CeDisGrem"),    # 1
    ("AdaDisGrem",     ada_disgrem,      (0.17, 0.44, 0.70), r"AdaDisGrem"),       # 2
    ("CeAdaDisGrem",   ce_ada_disgrem_adaptive, (0.03, 0.25, 0.53), r"CeAdaDisGrem"), # 3
    # ── Proposed quasi-Newton (AMBER accent) ──────────────────────────────
    ("DisGreQm",       dis_greqm,        (0.80, 0.52, 0.10), r"DisGre$\mathbb{Q}$m"),  # 4
    # ── First-order baselines (ORANGE family) ─────────────────────────────
    ("EXTRA",          extra,            (0.90, 0.62, 0.00), r"EXTRA"),            # 5
    ("DIGing",         diging,           (0.68, 0.44, 0.00), r"DIGing"),           # 6
    # ── Second-order baselines (WARM RED / BROWN family, light → dark) ────
    ("DQM",            dqm,              (0.84, 0.37, 0.00), r"DQM"),             # 7
    ("ESOM",           esom,             (0.75, 0.28, 0.30), r"ESOM"),            # 8
    # ── Quasi-Newton baseline (same warm family) ──────────────────────────
    ("DisQN",          dis_qn,           (0.60, 0.22, 0.42), r"DisQN"),           # 9
    # ── Recent second-order baselines ─────────────────────────────────────
    ("SONATA",         sonata,           (0.50, 0.16, 0.30), r"SONATA"),          # 10
    ("NetworkGIANT",   network_giant,    (0.38, 0.12, 0.22), r"Net-GIANT"),       # 11
]

# ── Mode map ───────────────────────────────────────────────────────────────
_MODE_MAP = {
    # ── PRIMARY COMPARISON ─────────────────────────────────────────────────
    "MainComp":      [0, 1, 2, 3, 5, 6, 7, 8, 10, 11],
    "MainCompCore":  [0, 1, 2, 3, 5, 6, 7, 8],
    "CEStudy":       [0, 1, 2, 3],
    "CommComp":      [0, 1, 2, 3, 5, 6, 7, 8],
    # ── FULL / SUPPLEMENTARY ──────────────────────────────────────────────
    "PaperFull":     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "PaperSecond":   [0, 1, 2, 3, 7, 8, 10, 11],
    "PaperMain":     [0, 1, 2, 3, 4, 5, 6],
    # ── THEMATIC SUBSETS ──────────────────────────────────────────────────
    "All":            list(range(len(_REGISTRY))),
    "DisGremFamily":  [0, 1, 2, 3, 4],
    "SecondOrder":    [7, 8, 10, 11],
    "RecentSecond":   [10, 11],
    "FirstOrder":     [5, 6],
    "ScaleComp":      [0, 1, 5, 6],
    "NewBaselines":   [5, 6],
    "OurVsFirst":     [0, 1, 4, 5, 6],
    "OurVsSecond":    [0, 1, 7, 8, 10, 11],
    "CeGroup":        [1, 3],
    "AdaGroup":       [2, 3],
    "DisGremNoAda":   [0, 1],
    "ScaleStudy":     [0, 2, 5, 10],
}


def get_alg_bank(test_key: str = "All") -> list:
    """Return list of (name, callable) tuples for the given test group."""
    if test_key not in _MODE_MAP:
        raise ValueError(f"Unknown test_key '{test_key}'. "
                         f"Available: {list(_MODE_MAP.keys())}")
    idx_list = _MODE_MAP[test_key]
    return [(_REGISTRY[i][0], _REGISTRY[i][1]) for i in idx_list]


def get_alg_style(name: str) -> tuple:
    """Return (label, color, linestyle) for an algorithm by name."""
    # Line styles: our algs use solid/dashed, baselines use dotted/dash-dot
    styles = {
        # Ours (blue): vary line style for internal distinction
        "DisGrem":        "-",
        "CeDisGrem":      "--",
        "AdaDisGrem":     "-.",
        "CeAdaDisGrem":   (0, (1, 1)),          # densely dotted
        "DisGreQm":       (0, (3, 1, 1, 1)),    # dash-dot-dot (amber)
        # First-order (green)
        "EXTRA":          "-",
        "DIGing":         "--",
        # Second-order baselines (red/warm)
        "DQM":            "-",
        "ESOM":           "--",
        "DisQN":          "-.",
        "SONATA":         (0, (5, 2)),           # loosely dashed
        "NetworkGIANT":   (0, (3, 1, 1, 1)),    # dash-dot-dot
    }
    for entry in _REGISTRY:
        n, _, color, label = entry[0], entry[1], entry[2], entry[3]
        if n == name:
            return label, color, styles.get(name, "-")
    return name, (0.5, 0.5, 0.5), "-"


def get_alg_groups() -> dict:
    """
    Return algorithm groupings used in plot styling and paper tables.

    Groups
    ------
    ours_second : proposed second-order algorithms
    ours_quasi  : proposed quasi-Newton algorithm
    ours        : all proposed algorithms (union of above)
    first       : first-order baselines
    second      : second-order baselines
    quasi       : quasi-Newton baseline
    """
    return {
        "ours_second": ["DisGrem", "CeDisGrem", "AdaDisGrem", "CeAdaDisGrem"],
        "ours_quasi":  ["DisGreQm"],
        "ours":        ["DisGrem", "CeDisGrem", "AdaDisGrem", "CeAdaDisGrem", "DisGreQm"],
        "first":       ["EXTRA", "DIGing"],
        "second":      ["DQM", "ESOM", "SONATA", "NetworkGIANT"],
        "second_classic": ["DQM", "ESOM"],
        "second_recent":  ["SONATA", "NetworkGIANT"],
        "quasi":       ["DisQN"],
    }
