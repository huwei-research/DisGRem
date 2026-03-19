"""
compress.py – Matrix compression utilities
Ported from MATLAB: compress.m, compress_topk.m, compress_lowrank.m
"""

from __future__ import annotations
import numpy as np


def compress_topk(H: np.ndarray, k: int) -> np.ndarray:
    """
    Top-k sparsification for a symmetric matrix.
    Keep the k entries with largest absolute value (upper triangle + diag),
    then symmetrise.
    """
    d = H.shape[0]
    Hc = np.zeros((d, d))

    # indices of upper triangle (including diagonal)
    rows, cols = np.triu_indices(d)
    vals = H[rows, cols]
    k_eff = min(k, len(vals))
    order = np.argsort(np.abs(vals))[::-1][:k_eff]
    keep_r = rows[order]
    keep_c = cols[order]

    Hc[keep_r, keep_c] = H[keep_r, keep_c]
    # symmetrise
    Hc = Hc + np.triu(Hc, 1).T
    return Hc


def compress_lowrank(H: np.ndarray, r: int) -> np.ndarray:
    """
    Rank-r truncated SVD approximation of a symmetric matrix.
    """
    r = min(r, H.shape[0])
    U, s, _ = np.linalg.svd((H + H.T) / 2)
    return (U[:, :r] * s[:r]) @ U[:, :r].T


def compress(H: np.ndarray, method: str, param: float) -> np.ndarray:
    """
    Dispatcher for matrix compression.

    Parameters
    ----------
    H      : symmetric (d×d) matrix
    method : 'topk' | 'lowrank'
    param  : k for topk, rank r for lowrank
    """
    H = (H + H.T) / 2          # enforce symmetry
    method = method.lower()

    if method == "topk":
        return compress_topk(H, int(param))
    elif method == "lowrank":
        return compress_lowrank(H, int(param))
    else:
        raise ValueError(f"compress: unknown method '{method}'")
