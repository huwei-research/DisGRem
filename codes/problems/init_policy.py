"""
init_policy.py – Centralised parameter & initial-point policy.
Ported from MATLAB: init_policy.m

Each entry in M_alpha_policy may contain:
    M_factor  : scalar multiplier of L_max for M
    alpha     : step size numerator (actual alpha = alpha / L_max)
    decay     : bool – whether to use α/√k decay
    maxIt     : per-function iteration limit (overrides global P["maxIt"])
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


def init_policy(mode: str = "regular") -> Tuple[dict, dict, dict]:
    """
    Parameters
    ----------
    mode : 'regular' | 'robust' | 'parameter'

    Returns
    -------
    param_bank      : dict – obj_factory arguments per function name
    M_alpha_policy  : dict (or list for 'parameter') – {M_factor, alpha, decay, maxIt}
    x0_generator    : dict – callables (d, far) → x0
    """
    assert mode in ("regular", "robust", "parameter"), f"Unknown mode: {mode}"
    d_default = 30

    # ── shared x0 helpers ──────────────────────────────────────────────────
    def ridge_x0(d, far):
        if far:
            return 10 * np.random.randn(d)
        return np.random.rand(d)

    def _case(far, near_fn, far_fn):
        return far_fn() if far else near_fn()

    if mode in ("regular", "robust"):
        # ── param_bank ─────────────────────────────────────────────────────
        param_bank = {
            "ridge":                [d_default, 1e-3],
            "quadbad":              [d_default, 1e3],
            "huber":                [d_default],
            "logsumexp":            [d_default],
            "linlog":               [d_default],
            "rosenbrock":           [d_default],
            "styblinski_tang":      [d_default],
            "logreg_real":          ["svmguide3", 0.01],
            "logreg_ncvr":          ["svmguide3", 0.05],
        }

        # ── M–alpha–maxIt policy ────────────────────────────────────────────
        # alpha is the numerator: actual step size = alpha / L_max
        # maxIt overrides global setting; omitting means use global default
        # Notes on tolType:
        #   "combo" : stop when gradNrm + cons < tol  (default; best for convex)
        #   "relF"  : stop when |F-F*|/|F0-F*| < tol  (best for non-convex; avoids
        #             floating-point gradient noise at machine precision)
        M_alpha_policy = {
            # ── Convex problems ──────────────────────────────────────────────
            "ridge":           {"M_factor": 0.1,  "alpha": 0.20, "decay": False,
                                "maxIt": 200,  "tolType": "combo"},
            "huber":           {"M_factor": 1.5,  "alpha": 0.30, "decay": False,
                                "maxIt": 800,  "tolType": "combo"},
            "logsumexp":       {"M_factor": 5.0,  "alpha": 0.30, "decay": False,
                                "maxIt": 400,  "tolType": "combo"},
            "linlog":          {"M_factor": 1.0,  "alpha": 0.20, "decay": False,
                                "maxIt": 1500, "tolType": "relF"},
            # ── Ill-conditioned convex ───────────────────────────────────────
            "quadbad":         {"M_factor": 0.1,  "alpha": 0.10, "decay": False,
                                "maxIt": 1500, "tolType": "relF"},
            # ── Real-data convex / non-convex ────────────────────────────────
            "logreg_real":     {"M_factor": 3.0,  "alpha": 1.0,  "decay": False,
                                "maxIt": 600,  "tolType": "combo"},
            "logreg_ncvr":     {"M_factor": 3.0,  "alpha": 1.0,  "decay": True,
                                "maxIt": 1000, "tolType": "relF"},
            # ── Non-convex problems ──────────────────────────────────────────
            "rosenbrock":      {"M_factor": 3.0,  "alpha": 0.10, "decay": True,
                                "maxIt": 300,  "tolType": "relF"},
            "styblinski_tang": {"M_factor": 15.0, "alpha": 0.05, "decay": True,
                                "maxIt": 100,  "tolType": "relF"},
        }

        # ── x0 generator ───────────────────────────────────────────────────
        x0_generator = {
            "ridge":           ridge_x0,
            "quadbad":         ridge_x0,
            "huber":           ridge_x0,
            "logsumexp":       lambda d, far: _case(far,
                                    lambda: np.log(1 + np.random.rand(d)),
                                    lambda: 5 + np.random.rand(d)),
            "linlog":          lambda d, far: 3 * np.random.randn(d) if far else np.random.rand(d),
            "rosenbrock":      lambda d, far: _case(far,
                                    lambda: -0.5 * np.ones(d),
                                    lambda: -5 * np.ones(d)),
            "styblinski_tang": lambda d, far: _case(far,
                                    lambda: 0.5 * np.random.rand(d) - 2,
                                    lambda: 20 * np.random.rand(d) - 10),
            "logreg_real":          ridge_x0,
            "logreg_ncvr":          ridge_x0,
        }

        return param_bank, M_alpha_policy, x0_generator

    else:  # 'parameter'
        # ── parameter-sweep policy (5 × 4 = 20 configurations) ─────────────
        param_bank = {"ridge": [d_default, 1e-3]}

        M_alpha_list = []
        for m_val in [0.05, 0.1, 0.5, 1.0, 3.0]:
            M_alpha_list.append({
                "M_factor": float(m_val),
                "alpha":    1.0,
                "decay":    False,
                "name":     f"M={m_val}L",
            })

        x0_generator = {"ridge": ridge_x0}
        return param_bank, M_alpha_list, x0_generator
