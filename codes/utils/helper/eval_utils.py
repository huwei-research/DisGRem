"""
eval_utils.py - approx_grad, approx_hess, compute_info, eval_all, grad_hess_all
Ported from MATLAB: approx_grad.m, approx_hess.m, compute_info.m, eval_all.m
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple, List
import numpy as np


# ─────────────────────────────────────────────────────────────
#  approx_grad  (central finite difference)
# ─────────────────────────────────────────────────────────────
def approx_grad(fun: Callable, x: np.ndarray) -> np.ndarray:
    """
    Approximate gradient via central finite differences.
    g[j] = (f(x + h*e_j) - f(x - h*e_j)) / (2h)
    """
    x = x.ravel()
    d = len(x)
    g = np.zeros(d)
    h = 1e-6 * max(1.0, float(np.linalg.norm(x)))

    for j in range(d):
        e = np.zeros(d)
        e[j] = 1.0
        g[j] = (fun(x + h * e) - fun(x - h * e)) / (2 * h)
    return g


# ─────────────────────────────────────────────────────────────
#  approx_hess  (central finite difference)
# ─────────────────────────────────────────────────────────────
def approx_hess(fun: Callable, x: np.ndarray) -> np.ndarray:
    """
    Approximate Hessian via central finite differences.
    """
    x = x.ravel()
    d = len(x)
    H = np.zeros((d, d))
    h = 1e-6 * max(1.0, float(np.linalg.norm(x)))

    try:
        fx = float(fun(x))
        if not np.isfinite(fx):
            return np.zeros((d, d))
    except Exception:
        return np.zeros((d, d))

    for i in range(d):
        ei = np.zeros(d); ei[i] = 1.0
        for j in range(i, d):
            ej = np.zeros(d); ej[j] = 1.0
            try:
                if i == j:
                    fpp = fun(x + h * ei)
                    fmm = fun(x - h * ei)
                    H[i, i] = (fpp - 2 * fx + fmm) / h ** 2
                else:
                    fpp = fun(x + h * ei + h * ej)
                    fpm = fun(x + h * ei - h * ej)
                    fmp = fun(x - h * ei + h * ej)
                    fmm = fun(x - h * ei - h * ej)
                    val = (fpp - fpm - fmp + fmm) / (4 * h ** 2)
                    H[i, j] = val
                    H[j, i] = val
            except Exception:
                return np.zeros((d, d))
    return H


# ─────────────────────────────────────────────────────────────
#  eval_all
# ─────────────────────────────────────────────────────────────
def eval_all(f_list: List[Callable], X: np.ndarray) -> np.ndarray:
    """
    Evaluate f_list[i](X[:, i]) for all agents.

    Parameters
    ----------
    f_list : list of N callables
    X      : (d, N) matrix

    Returns
    -------
    vals : (N,) array
    """
    N = X.shape[1]
    return np.array([f_list[i](X[:, i]) for i in range(N)])


# ─────────────────────────────────────────────────────────────
#  grad_hess_all  (analytical, from fname + fparam)
# ─────────────────────────────────────────────────────────────
def grad_hess_all(x: np.ndarray, fname: str,
                  fparam: dict, mode: str = "both"):
    """
    Compute analytical gradient / Hessian for known function types.

    Supported fname: 'ridge', 'quadbad', 'logsumexp', 'huber',
                     'linlog', 'logreg_real', 'logreg_ncvr',
                     'rosenbrock', 'styblinski_tang'

    Returns
    -------
    g : (d,) array or None
    H : (d,d) array or None
    """
    x = x.ravel()
    d = len(x)
    g, H = None, None

    fname = fname.lower()

    if fname == "ridge":
        A = fparam["A"]
        y = fparam["y"]
        lam = fparam["lambda"]
        r = A @ x - y
        if mode in ("grad", "both"):
            g = A.T @ r + lam * x
        if mode in ("hess", "both"):
            H = A.T @ A + lam * np.eye(d)

    elif fname == "quadbad":
        Q = fparam["Q"]
        b = fparam["b"]
        if mode in ("grad", "both"):
            g = Q @ x + b
        if mode in ("hess", "both"):
            H = Q.copy()

    elif fname == "logsumexp":
        A = fparam["A"]   # (d, p)
        b = fparam["b"]   # (p,)
        rho = fparam["rho"]
        z = A.T @ x - b           # (p,)
        z_shift = z - z.max()
        e = np.exp(z_shift / rho)
        p_vec = e / e.sum()        # softmax
        if mode in ("grad", "both"):
            g = A @ p_vec
        if mode in ("hess", "both"):
            Ap = A @ p_vec                                         # (d,)
            H = ((A * p_vec[None, :]) @ A.T - np.outer(Ap, Ap)) / rho

    elif fname == "huber":
        A = fparam["A"]    # (p, d)
        b = fparam["b"]    # (p,)
        delta = fparam.get("delta", 1.0)
        r = A @ x - b
        s2 = 1.0 + (r / delta) ** 2          # (p,)
        sqrt_s2 = np.sqrt(s2)
        if mode in ("grad", "both"):
            g = A.T @ (r / sqrt_s2)
        if mode in ("hess", "both"):
            d2phi = 1.0 / (s2 * sqrt_s2)     # = 1/(1+(r/delta)^2)^{3/2}
            H = A.T @ (d2phi[:, None] * A)

    elif fname == "linlog":
        A = fparam["A"]    # (d, d)
        b = fparam["b"]    # (d,)
        r = A @ x - b
        absr = np.abs(r)
        if mode in ("grad", "both"):
            dphidr = np.where(absr <= 1, r,
                              np.sign(r) / np.maximum(absr, 1e-300))
            g = A.T @ dphidr
        if mode in ("hess", "both"):
            d2phi = np.where(absr <= 1, 1.0, -1.0 / np.maximum(absr ** 2, 1e-300))
            H = A.T @ (d2phi[:, None] * A)

    elif fname in ("logreg_real", "logreg_ncvr"):
        A_i = fparam["A"]    # (m_i, d)
        b_i = fparam["b"]    # (m_i,) labels ±1
        iota = fparam.get("iota", 0.0)
        alpha = fparam.get("alpha", 0.0)
        m_i = A_i.shape[0]

        z = b_i * (A_i @ x)
        # sigmoid = 1 / (1 + exp(-z * b))  → p = sigmoid(-z)
        sig = _sigmoid(-z)   # probability of misclassification

        if mode in ("grad", "both"):
            g = -A_i.T @ (b_i * sig) / m_i + iota * x
            if fname == "logreg_ncvr":
                # nonconvex reg: R_ncvr'(x_j) = 2*alpha*x_j / (1+x_j^2)^2
                g = g + 2 * alpha * x / (1 + x ** 2) ** 2
        if mode in ("hess", "both"):
            w = sig * (1 - sig) / m_i
            H = A_i.T @ (w[:, None] * A_i) + iota * np.eye(d)
            if fname == "logreg_ncvr":
                # R_ncvr''(x_j) = 2*alpha*(1 - 3*x_j^2) / (1 + x_j^2)^3
                denom3 = np.maximum((1 + x ** 2) ** 3, 1e-30)
                diag_r = 2 * alpha * (1 - 3 * x ** 2) / denom3
                H += np.diag(diag_r)

    elif fname == "rosenbrock":
        # f = Σ_{k} 100(x_{2k+1} - x_{2k}^2)^2 + (x_{2k} - 1)^2
        xo = x[::2]    # even-indexed
        xe = x[1::2]   # odd-indexed
        r = xe - xo ** 2
        if mode in ("grad", "both"):
            g = np.zeros(d)
            g[::2]  = -400.0 * xo * r + 2.0 * (xo - 1.0)
            g[1::2] = 200.0 * r
        if mode in ("hess", "both"):
            H = np.zeros((d, d))
            idx_e = np.arange(0, d, 2)
            idx_o = np.arange(1, d, 2)
            H[idx_e, idx_e] = 1200.0 * xo ** 2 - 400.0 * xe + 2.0
            H[idx_o, idx_o] = 200.0
            H[idx_e, idx_o] = -400.0 * xo
            H[idx_o, idx_e] = -400.0 * xo

    elif fname == "styblinski_tang":
        # f = Σ_i (x_i⁴ - 16x_i^2 + 5x_i)
        if mode in ("grad", "both"):
            g = 4.0 * x ** 3 - 32.0 * x + 5.0
        if mode in ("hess", "both"):
            H = np.diag(12.0 * x ** 2 - 32.0)

    else:
        return None, None

    return g, H


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid - no overflow warnings."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    exp_neg = np.exp(-z[pos])
    out[pos] = 1.0 / (1.0 + exp_neg)
    exp_pos = np.exp(z[~pos])
    out[~pos] = exp_pos / (1.0 + exp_pos)
    return out


# ─────────────────────────────────────────────────────────────
#  compute_info  (main interface used by all algorithms)
# ─────────────────────────────────────────────────────────────
def compute_info(x: np.ndarray, prm: dict, i: int,
                 mode: str = "both") -> Tuple[Optional[np.ndarray],
                                              Optional[np.ndarray]]:
    """
    Retrieve gradient and / or Hessian for agent i at point x.

    prm.info == 0  → finite differences
    prm.info >= 1  → analytical (grad_hess_all), fall back to FD if not available
    """
    x = x.ravel()
    g, H = None, None

    if prm.get("info", 0) >= 1:
        fname = prm.get("fname", "")
        fp_list = prm.get("fparam", [])
        # Guard: if fparam is shorter than N, fall back to FD for this agent
        fparam_i = fp_list[i] if i < len(fp_list) else {}
        g_a, H_a = grad_hess_all(x, fname, fparam_i, mode)
        if mode in ("grad", "both") and g_a is not None:
            g = g_a
        if mode in ("hess", "both") and H_a is not None:
            H = H_a

    # Fall back to FD where needed
    f_i = prm["f"][i]
    if mode in ("grad", "both") and g is None:
        g = approx_grad(f_i, x)
    if mode in ("hess", "both") and H is None:
        try:
            H = approx_hess(f_i, x)
        except Exception:
            H = np.zeros((len(x), len(x)))

    # Single-return alias
    if mode == "grad":
        return g, None
    if mode == "hess":
        return None, H
    return g, H
