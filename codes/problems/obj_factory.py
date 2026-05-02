"""
obj_factory.py - Build heterogeneous local objective functions.

Supported types (9):
    Convex:      'ridge', 'quadbad', 'logsumexp', 'huber', 'linlog', 'logreg_real'
    Non-convex:  'rosenbrock', 'styblinski_tang', 'logreg_ncvr'
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Callable
import numpy as np
from scipy.optimize import minimize


def _find_consensus_opt(fun_list, d, x0=None, n_restarts=2):
    """
    Numerically find the consensus minimum of F(x) = mean_i f_i(x) using L-BFGS-B.
    Tries n_restarts starting points (x=0 plus random) and returns the best solution.
    For convex functions, a single start from x=0 typically suffices.
    """
    def F(x):
        return float(np.mean([fi(x) for fi in fun_list]))

    best_x, best_f = None, np.inf
    starts = [np.zeros(d)] if x0 is None else [x0]
    if n_restarts > 1:
        rng = np.random.RandomState(0)
        for _ in range(n_restarts - 1):
            starts.append(rng.randn(d) * 0.5)

    for x_start in starts:
        try:
            res = minimize(F, x_start, method="L-BFGS-B",
                           options={"maxiter": 10000, "ftol": 1e-15, "gtol": 1e-12})
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except Exception:
            pass
    if best_x is None:
        best_x = np.zeros(d)
    return best_x, float(best_f)


def _log1pexp(x: np.ndarray) -> np.ndarray:
    """log(1 + exp(x)), numerically stable."""
    out = np.empty_like(x, dtype=float)
    pos = x > 0
    out[pos] = x[pos] + np.log1p(np.exp(-x[pos]))
    out[~pos] = np.log1p(np.exp(x[~pos]))
    return out


def _lse_stable(A: np.ndarray, x: np.ndarray,
                b: np.ndarray, rho: float) -> float:
    """rho * log(sum(exp((A^T x - b)/rho))), numerically stable.
    Correct form: z_max + rho*log(Σ), NOT rho*(z_max+log(Σ)).
    """
    z = A.T @ x - b
    z_max = z.max()
    return float(z_max + rho * np.log(np.sum(np.exp((z - z_max) / rho))))


def _pseudo_huber_vec(r: np.ndarray, delta: float) -> float:
    """Pseudo-Huber loss (Cinf smooth):  delta^2(√(1+(r/delta)^2) - 1), summed."""
    return float(delta ** 2 * np.sum(np.sqrt(1.0 + (r / delta) ** 2) - 1.0))


def _linlog_vec(r: np.ndarray) -> float:
    absr = np.abs(r)
    quad = absr <= 1
    return float(0.5 * np.sum(r[quad] ** 2)
                 + np.sum(np.log(np.maximum(absr[~quad], 1e-300)) + 0.5))


# ─────────────────────────────────────────────────────────────────────────────
#  Main factory
# ─────────────────────────────────────────────────────────────────────────────
def obj_factory(ftype: str, n_agent: int,
                *args, seed_offset: int = 0):
    """
    Parameters
    ----------
    ftype        : function family name (see docstring)
    n_agent      : N, number of agents
    *args        : positional arguments depending on ftype
                   (e.g. dim, lambda for 'ridge')
    seed_offset  : random seed shift

    Returns
    -------
    fun_list   : list of N callables f_i(x)
    dim        : problem dimension d
    L_vec      : (N,) Lipschitz constants
    x_opt_list : list of N x_opt (or list of one shared opt)
    f_opt_list : (N,) f_opt values
    is_convex  : (N,) bool array
    fname      : string (canonical name)
    fparam     : list of N dicts with per-agent parameters
    """
    def rng_i(i):
        np.random.seed(seed_offset + i)

    ftype_l = ftype.lower()

    # ── RIDGE ────────────────────────────────────────────────────────────────
    if ftype_l == "ridge":
        d = int(args[0]) if len(args) > 0 else 10
        lam = float(args[1]) if len(args) > 1 else 1e-3

        np.random.seed(0)
        x_true = np.random.rand(d)

        fun_list, fparam, L_vec, is_convex = [], [], np.zeros(n_agent), np.ones(n_agent, bool)
        H_sum = np.zeros((d, d))
        c_sum = np.zeros(d)   # accumulate A_i^T y_i (RHS of normal equations)

        for i in range(n_agent):
            rng_i(i + 1)
            m, mi = 30, 5
            A_i = np.random.rand(m * mi, d)
            y_i = A_i @ x_true + 0.05 * np.random.randn(m * mi)
            H_i = A_i.T @ A_i + lam * np.eye(d)
            H_sum += H_i
            c_sum += A_i.T @ y_i   # normal-equation RHS contribution
            fun_list.append(lambda x, Ai=A_i, yi=y_i:
                            0.5 * np.sum((Ai @ x - yi) ** 2) + 0.5 * lam * np.sum(x ** 2))
            L_vec[i] = float(np.linalg.eigvalsh(H_i).max())
            fparam.append({"A": A_i, "y": y_i, "lambda": lam})

        # Global consensus minimiser: H_bar x* = (1/N) sum_i A_i^T y_i
        # Correct formula: x* = H_bar^{-1} @ (c_sum / n_agent)
        H_bar = H_sum / n_agent
        x_star = np.linalg.solve(H_bar, c_sum / n_agent)
        f_star = float(np.mean([f(x_star) for f in fun_list]))
        return fun_list, d, L_vec, [x_star], np.full(n_agent, f_star), is_convex, "ridge", fparam

    # ── QUADBAD ──────────────────────────────────────────────────────────────
    elif ftype_l == "quadbad":
        d = int(args[0]) if len(args) > 0 else 10
        kappa = float(args[1]) if len(args) > 1 else 1e3

        fun_list, fparam, L_vec, is_convex = [], [], np.zeros(n_agent), np.ones(n_agent, bool)
        Q_sum = np.zeros((d, d)); b_sum = np.zeros(d)

        for i in range(n_agent):
            rng_i(i + 1)
            chi_i = kappa * 10 ** (2 * (np.random.rand() - 0.5))
            Q = np.diag(np.logspace(0, np.log10(chi_i), d))
            b = np.random.randn(d)
            fun_list.append(lambda x, Q=Q, b=b: float(0.5 * x @ Q @ x + b @ x))
            fparam.append({"Q": Q, "b": b})
            L_vec[i] = Q.diagonal().max()
            Q_sum += Q; b_sum += b

        # Consensus minimiser: (1/N) sum_i (Q_i x + b_i) = 0  →  x* = -Q_bar^{-1} b_bar
        Q_bar = Q_sum / n_agent
        x_star = np.linalg.solve(Q_bar, -b_sum / n_agent)
        f_star = float(np.mean([f(x_star) for f in fun_list]))
        return fun_list, d, L_vec, [x_star], np.full(n_agent, f_star), is_convex, "quadbad", fparam

    # ── LOGSUMEXP ─────────────────────────────────────────────────────────────
    elif ftype_l == "logsumexp":
        d = int(args[0]) if len(args) > 0 else 10
        p = max(d + 2, 12)   # p > d ensures each agent Hessian is full rank
        rho_fixed = 0.5      # fixed rho → well-conditioned (kappa~10 vs ~1400 for random rho)

        fun_list, fparam, L_vec, is_convex = [], [], np.zeros(n_agent), np.ones(n_agent, bool)

        for i in range(n_agent):
            rng_i(i + 1)
            A = np.random.randn(d, p) * 0.2
            b = np.random.randn(p)
            rho = rho_fixed
            fun_list.append(lambda x, A=A, b=b, rho=rho: _lse_stable(A, x, b, rho))
            fparam.append({"A": A, "b": b, "rho": rho})
            L_vec[i] = float(np.linalg.norm(A) ** 2 / (4 * rho))

        x_star, f_star = _find_consensus_opt(fun_list, d)
        return (fun_list, d, L_vec, [x_star], np.full(n_agent, f_star),
                is_convex, "logsumexp", fparam)

    # ── HUBER ─────────────────────────────────────────────────────────────────
    elif ftype_l == "huber":
        d = int(args[0]) if len(args) > 0 else 10
        p = 5; delta = 1.0

        fun_list, fparam, L_vec, is_convex = [], [], np.zeros(n_agent), np.ones(n_agent, bool)

        for i in range(n_agent):
            rng_i(i + 1)
            A = np.random.randn(p, d)
            b = np.random.randn(p)
            fun_list.append(lambda x, A=A, b=b: _pseudo_huber_vec(A @ x - b, delta))
            fparam.append({"A": A, "b": b, "delta": delta})
            L_vec[i] = float(np.linalg.norm(A, 2) ** 2)

        x_star, f_star = _find_consensus_opt(fun_list, d)
        return (fun_list, d, L_vec, [x_star], np.full(n_agent, f_star),
                is_convex, "huber", fparam)

    # ── LINLOG ─────────────────────────────────────────────────────────────────
    elif ftype_l == "linlog":
        d = int(args[0]) if len(args) > 0 else 10

        fun_list, fparam, L_vec, is_convex = [], [], np.zeros(n_agent), np.zeros(n_agent, bool)

        for i in range(n_agent):
            rng_i(i + 1)
            A = np.random.randn(d, d)
            b = np.random.randn(d)
            fun_list.append(lambda x, A=A, b=b: _linlog_vec(A @ x - b))
            fparam.append({"A": A, "b": b})
            L_vec[i] = 2.0 * float(np.linalg.norm(A, 2) ** 2)

        x_star, f_star = _find_consensus_opt(fun_list, d)
        return (fun_list, d, L_vec, [x_star], np.full(n_agent, f_star),
                is_convex, "linlog", fparam)

    # ── ROSENBROCK ────────────────────────────────────────────────────────────
    elif ftype_l == "rosenbrock":
        d = int(args[0]) if len(args) > 0 else 10

        fun_list, fparam, L_vec, is_convex = [], [], np.zeros(n_agent), np.zeros(n_agent, bool)
        x_opts, f_opts = [], []

        for i in range(n_agent):
            rng_i(i + 1)
            def _rosen(x, d=d):
                x = x.ravel()
                return float(np.sum(100.0 * (x[1::2] - x[::2] ** 2) ** 2
                                    + (x[::2] - 1.0) ** 2))
            fun_list.append(_rosen)
            fparam.append({})
            L_vec[i] = 100.0
            x_opts.append(np.ones(d))
            f_opts.append(0.0)

        return fun_list, d, L_vec, x_opts, np.array(f_opts), is_convex, "rosenbrock", fparam

    # ── STYBLINSKI-TANG ────────────────────────────────────────────────────────
    elif ftype_l in ("styblinski_tang", "st"):
        d = int(args[0]) if len(args) > 0 else 10
        x_star = -2.903534 * np.ones(d)

        fun_list, fparam, L_vec, is_convex = [], [], np.zeros(n_agent), np.zeros(n_agent, bool)
        x_opts, f_opts = [], []

        for i in range(n_agent):
            rng_i(i + 1)
            fun_list.append(lambda x: float(np.sum(x ** 4 - 16 * x ** 2 + 5 * x)))
            fparam.append({})
            L_vec[i] = 1000.0
            x_opts.append(x_star.copy())
            f_opts.append(fun_list[-1](x_star))

        return fun_list, d, L_vec, x_opts, np.array(f_opts), is_convex, "styblinski_tang", fparam

    # ── LOGREG_REAL ───────────────────────────────────────────────────────────
    elif ftype_l == "logreg_real":
        from utils.data.load_dataset import load_dataset  # lazy import
        ds_name = str(args[0]) if len(args) > 0 else "a9a"
        iota = float(args[1]) if len(args) > 1 else 1e-3
        dim_override = int(args[2]) if len(args) > 2 else None

        A_all, b_all = load_dataset(ds_name, standardize="zscore", label_style="pm1")
        m_tot, d_raw = A_all.shape
        d = d_raw if dim_override is None else min(dim_override, d_raw)
        if d < d_raw:
            A_all = A_all[:, :d]

        rng_split = np.random.RandomState(seed_offset + 9999)
        idx = rng_split.permutation(m_tot)
        blk = int(np.ceil(m_tot / n_agent))

        fun_list, fparam, L_vec, is_convex = [], [], np.zeros(n_agent), np.ones(n_agent, bool)
        x_opts, f_opts = [], []

        for i in range(n_agent):
            id_i = idx[i * blk: min((i + 1) * blk, m_tot)]
            A_i = A_all[id_i]; b_i = b_all[id_i]

            def _logreg(x, A=A_i, b=b_i):
                return float(iota * 0.5 * np.dot(x, x)
                             + np.mean(_log1pexp(-b * (A @ x))))

            fun_list.append(_logreg)
            fparam.append({"A": A_i, "b": b_i, "iota": iota})
            # L = iota + max_j ||a_j||^2 / (4 * m_i)  [correct per-sample bound]
            m_i = A_i.shape[0]
            L_vec[i] = iota + 0.25 * float(np.max(np.sum(A_i ** 2, axis=1))) / m_i

        x_star_lr, f_star_lr = _find_consensus_opt(fun_list, d, n_restarts=3)
        for i in range(n_agent):
            x_opts.append(x_star_lr.copy())
            f_opts.append(fun_list[i](x_star_lr))

        return fun_list, d, L_vec, x_opts, np.array(f_opts), is_convex, "logreg_real", fparam

    # ── LOGREG_NCVR ───────────────────────────────────────────────────────────
    elif ftype_l == "logreg_ncvr":
        from utils.data.load_dataset import load_dataset
        ds_name = str(args[0]) if len(args) > 0 else "a9a"
        alpha = float(args[1]) if len(args) > 1 else 0.05
        dim_override = int(args[2]) if len(args) > 2 else None

        A_all, b_all = load_dataset(ds_name, standardize="zscore", label_style="pm1")
        m_tot, d_raw = A_all.shape
        d = d_raw if dim_override is None else min(dim_override, d_raw)
        if d < d_raw:
            A_all = A_all[:, :d]

        rng_split = np.random.RandomState(seed_offset + 9999)
        idx = rng_split.permutation(m_tot)
        blk = int(np.ceil(m_tot / n_agent))

        fun_list, fparam, L_vec, is_convex = [], [], np.zeros(n_agent), np.zeros(n_agent, bool)
        x_opts, f_opts = [], []

        for i in range(n_agent):
            id_i = idx[i * blk: min((i + 1) * blk, m_tot)]
            A_i = A_all[id_i]; b_i = b_all[id_i]

            def _ncvr(x, A=A_i, b=b_i):
                return float(np.mean(_log1pexp(-b * (A @ x)))
                             + alpha * np.sum(x ** 2 / (1 + x ** 2)))

            fun_list.append(_ncvr)
            fparam.append({"A": A_i, "b": b_i, "alpha": alpha})
            # L = max_j ||a_j||^2 / (4 * m_i) + alpha  [correct per-sample bound]
            m_i = A_i.shape[0]
            L_vec[i] = 0.25 * float(np.max(np.sum(A_i ** 2, axis=1))) / m_i + alpha

        x_star_nc, f_star_nc = _find_consensus_opt(fun_list, d, n_restarts=3)
        for i in range(n_agent):
            x_opts.append(x_star_nc.copy())
            f_opts.append(fun_list[i](x_star_nc))

        return fun_list, d, L_vec, x_opts, np.array(f_opts), is_convex, "logreg_ncvr", fparam

    else:
        raise ValueError(f"obj_factory: unknown type '{ftype}'")
