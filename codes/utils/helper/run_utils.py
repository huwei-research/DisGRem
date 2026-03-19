"""
run_utils.py – Shared utilities for experiment run scripts.

Centralises divergence detection, figure-directory creation, and other
helpers that were previously duplicated across run_regular / run_comm /
run_quasi / run_robust.
"""

from __future__ import annotations
import os
import numpy as np


# ── Divergence detection ─────────────────────────────────────────────────────
_DIVERGE_F_RATIO = 1e6
_DIVERGE_ABS     = 1e12


def detect_diverged(out: dict, f0_val: float = 0.0) -> bool:
    """
    Return True when the run is considered diverged / numerically failed.

    Catches:
      1. Exception / explicit failure flag
      2. NaN or Inf in the objective trace
      3. Objective explosion (final >> initial by large factor or absolute threshold)
      4. RelF explosion (final relF > 1.5 means objective WORSE than start)
      5. Total stagnation: combo hasn't improved by >1 % over last 100 steps
         AND is far from convergence (> 1e-4)
    """
    if out.get("fail") or out.get("__failed__"):
        return True
    vf = np.asarray(out.get("ValueF", []), dtype=float).ravel()
    if len(vf) == 0:
        return True
    if np.any(~np.isfinite(vf)):
        return True

    f0 = abs(float(vf[0])) if np.isfinite(vf[0]) else abs(f0_val)
    ref = max(f0, 1.0)
    if float(vf[-1]) > max(ref * _DIVERGE_F_RATIO, _DIVERGE_ABS):
        return True

    relF_arr = np.asarray(out.get("relF", []), dtype=float).ravel()
    if len(relF_arr) > 0:
        final_relF = float(relF_arr[-1])
        if np.isfinite(final_relF) and final_relF > 1.5:
            return True

    combo_arr = np.asarray(out.get("combo", []), dtype=float).ravel()
    if len(combo_arr) >= 200:
        recent  = combo_arr[-100:]
        earlier = combo_arr[-200:-100]
        r_min = float(np.nanmin(recent[np.isfinite(recent)])) if np.any(np.isfinite(recent)) else np.nan
        e_min = float(np.nanmin(earlier[np.isfinite(earlier)])) if np.any(np.isfinite(earlier)) else np.nan
        if (np.isfinite(r_min) and np.isfinite(e_min) and
                r_min > 1e-4 and
                abs(r_min - e_min) / max(abs(e_min), 1e-15) < 0.01):
            return True

    return False


def ensure_fig_dirs(results_dir: str, *subdirs: str) -> None:
    """Create results_dir and all requested sub-directories."""
    os.makedirs(results_dir, exist_ok=True)
    for s in subdirs:
        os.makedirs(os.path.join(results_dir, s), exist_ok=True)
