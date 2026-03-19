"""
log_utils.py – init_log, trim_log, compute_comm_cost, printProgress
Ported from MATLAB: init_log.m, trim_log.m, compute_comm_cost.m, printProgress.m
"""

from __future__ import annotations
import numpy as np
import time


# ─────────────────────────────────────────────────────────────
#  init_log
# ─────────────────────────────────────────────────────────────
def init_log(T: int) -> dict:
    """
    Preallocate a logging dict for T iterations.
    Each numeric metric is initialised to NaN(T,).
    """
    log = {
        "ValueF":   np.full(T, np.nan),
        "gradNrm":  np.full(T, np.nan),
        "cons":     np.full(T, np.nan),
        "combo":    np.full(T, np.nan),
        "relX":     np.full(T, np.nan),
        "relF":     np.full(T, np.nan),
        "rk":       np.full(T, np.nan),
        "commCost": np.full(T, np.nan),
        "timeCost": np.full(T, np.nan),
        "Mavg":     np.full(T, np.nan),
    }
    return log


# ─────────────────────────────────────────────────────────────
#  trim_log
# ─────────────────────────────────────────────────────────────
def trim_log(log: dict, k: int) -> dict:
    """
    Truncate all vector fields to length min(k, len(field)).
    """
    for key, val in log.items():
        if isinstance(val, np.ndarray):
            log[key] = val[:min(k, len(val))]
    return log


# ─────────────────────────────────────────────────────────────
#  compute_comm_cost
# ─────────────────────────────────────────────────────────────
def compute_comm_cost(d: int, W: np.ndarray,
                      method: str = "vector",
                      param: float = None,
                      rounds: int = 1) -> float:
    """
    Estimate total communication cost in bytes.

    Parameters
    ----------
    d       : dimension of vector / matrix
    W       : N×N mixing (adjacency) matrix
    method  : 'vector' | 'full_matrix' | 'topk' | 'lowrank'
    param   : compression parameter (k for topk, r for lowrank)
    rounds  : number of gossip rounds

    Returns
    -------
    cost    : bytes (float)
    """
    N = W.shape[0]
    # Count directed edges by zeroing the diagonal explicitly, avoiding
    # assumption that all N diagonal entries are nonzero
    W_offdiag = W - np.diag(np.diag(W))
    n_edges = int(np.count_nonzero(W_offdiag))

    method = method.lower()
    if method == "vector":
        cost_per_agent = 8 * d
    elif method == "full_matrix":
        cost_per_agent = 8 * d * (d + 1) // 2
    elif method == "topk":
        k = int(param) if param is not None else min(10, d * d)
        cost_per_agent = 8 * k + 8 * k
    elif method == "lowrank":
        t = int(param) if param is not None else min(2, d)
        cost_per_agent = 8 * (d * t + t * t)
    else:
        raise ValueError(f"Unknown method: {method}")

    return cost_per_agent * n_edges * rounds


# ─────────────────────────────────────────────────────────────
#  printProgress
# ─────────────────────────────────────────────────────────────
def print_progress(k: int, grad_nrm: float, cons: float,
                   value_f: float, **kwargs) -> None:
    """
    Print formatted progress line, mirroring MATLAB printProgress.m
    """
    msg = (f"[iter {k:5d}]  gradNrm={grad_nrm:.3e}  cons={cons:.3e}"
           f"  f={value_f:.6e}")
    for key, val in kwargs.items():
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            msg += f"  {key}={val:.3e}" if isinstance(val, (int, float)) else f"  {key}={val}"
    print(msg)
