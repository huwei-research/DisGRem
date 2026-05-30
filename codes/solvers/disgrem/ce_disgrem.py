"""
ce_disgrem.py - CeDisGrem: Communication-Efficient Decentralised
Gradient-Regularised Newton with Gradient & Hessian Tracking.

Aligned with Algorithm 1 (DisGrem) in the paper:
  (A) Pre-mixing:  x̃ = W^tau x,  g̃ = W^tau g,  H̃ = W^tau H
  (B) Newton step:  sym(H̃) + (lambda+delta)I,  delta = max{0, -lambda_min(sym(H̃))}
  (C) Post-mixing:  x_{k+1} = W^t y
  (D) Trackers:    g_{k+1} = W^t(g̃ + gradf_new - gradf_old)
                   H_{k+1} = sym(W^t(H̃ + grad^2f_new - grad^2f_old))

Key notation (all arrays are (d, N)):
    X       : local iterates
    G       : gradient tracker
    H       : Hessian tracker  (d, d, N)
    gx_stack: vectorised G for mixing  (d*N,)
    Hstack  : vectorised H for mixing  (d*N, d)
"""

from __future__ import annotations
import time
import numpy as np
from utils.helper.log_utils import init_log, trim_log, compute_comm_cost, print_progress
from utils.helper.eval_utils import compute_info
from utils.helper.compress import compress
from utils.helper.graph import get_W


def _mix_vec(W, v, d, N):
    """(W kron I_d) @ v  without forming the Kronecker product.  O(dN^2)."""
    return (v.reshape(d, N, order='F') @ W.T).ravel(order='F')


def _mix_mat(W, M, d, N):
    """(W kron I_d) @ M  for M of shape (dN, cols).  O(dN^2 * cols)."""
    cols = M.shape[1]
    M3 = M.reshape(d, N, cols, order='F')
    return (M3.transpose(0, 2, 1) @ W.T).transpose(0, 2, 1).reshape(
        d * N, cols, order='F')


def _mixing_rho_from_W(W):
    """Return ||W-J||_2 clipped to (0,1) for the current mixing matrix."""
    W_arr = np.asarray(W, dtype=float)
    n = W_arr.shape[0]
    J = np.ones((n, n), dtype=float) / float(n)
    rho = float(np.linalg.norm(W_arr - J, 2))
    return min(max(rho, 1e-12), 1.0 - 1e-12)


def _log_consensus_rounds(prm, k, Wk):
    """Logarithmic schedule ceil((p log(k+2)+c_mix)/(-log rho)), limited by NC_max."""
    if prm.get("NC_schedule", "fixed") != "log":
        return int(prm.get("NC", 1))
    p_log = float(prm.get("log_p", 3.0))
    c_mix = float(prm.get("log_c_mix", 2.0))
    rho = prm.get("_log_rho_cached")
    if rho is None:
        rho = _mixing_rho_from_W(Wk)
        prm["_log_rho_cached"] = rho
    rounds = int(np.ceil((p_log * np.log(k + 2.0) + c_mix) / (-np.log(rho))))
    nc_max = int(prm.get("NC_max", 999))
    return max(1, min(rounds, nc_max))


def ce_disgrem(x0: np.ndarray, prm: dict):
    """
    CeDisGrem algorithm.

    Parameters
    ----------
    x0  : (d,) initial point
    prm : parameter dict with fields:
          .f          list[N] of callables
          .W          (N,N) ndarray or callable(k) -> (N,N)
          .Nagent     N
          .dim        d
          .M          regularisation constant
          .maxIt      max iterations         [200]
          .tol        stopping threshold     [1e-16]
          .tolType    'combo'|'relX'|'relF'  ['combo']
          .verbose    bool                   [True]
          .NC         consensus rounds       [1]
          .Klazy      Hessian-tracking period[maxIt//5]
          .compressH  bool                   [True]
          .compressor 'topk'|'lowrank'       ['topk']
          .compressParam int                 [0.10*d^2]
          .info       0|1|2                  [0]
          .x_opt, .f_opt  optional optima

    Returns
    -------
    x_all   : (d*N,) stacked final iterates
    log_data: dict with per-iteration metrics
    """
    # ── defaults ───────────────────────────────────────────────────────────
    p = {
        "maxIt": 200, "tol": 1e-16, "tolType": "combo",
        "verbose": True, "M": 10.0,
        "compressH": True, "compressor": "topk", "compressParam": None,
        "NC": 3, "Klazy": None, "countComm": True,
        "memoryLimitMB": np.inf, "info": 0,
    }
    p.update(prm)
    prm = p

    N = prm["Nagent"]
    d = prm["dim"]
    f = prm["f"]
    K = prm["maxIt"]
    print_freq = max(1, K // 25)

    # ── parameter validation ────────────────────────────────────────────────
    assert N == len(f), f"Nagent={N} but len(f)={len(f)}"
    assert d == len(x0.ravel()), f"dim={d} but x0 has length {len(x0.ravel())}"
    assert prm.get("W") is not None, "Mixing matrix W is required"
    assert prm["M"] > 0, f"Regularisation M must be positive, got {prm['M']}"

    if prm["compressParam"] is None:
        prm["compressParam"] = max(1, min(round(0.10 * d ** 2), int(1e5)))
    if prm["Klazy"] is None:
        prm["Klazy"] = max(1, min(K // 4, 80))

    # ── allocate state ──────────────────────────────────────────────────────
    X = np.tile(x0.ravel()[:, None], (1, N))   # (d, N)
    G = np.zeros((d, N))
    H = np.zeros((d, d, N))
    gx_stack = np.zeros(d * N)
    Hstack = np.zeros((d * N, d))

    # ── logging ─────────────────────────────────────────────────────────────
    log_data = init_log(K)
    log_data["NC"] = np.full(K, np.nan)
    _ace_klazy0 = prm.get("Klazy") or max(1, min(K // 4, 80))
    _ace_cp0 = prm.get("compressParam") or max(1, min(round(0.10 * d ** 2), int(1e5)))
    _ace_nc0 = max(1, prm.get("NC", 3))
    log_data["xBar"] = np.full((K, d), np.nan)

    # ── initial grad / Hessian ──────────────────────────────────────────────
    for i in range(N):
        gi, Hi = compute_info(X[:, i], prm, i, "both")
        G[:, i] = gi
        H[:, :, i] = Hi

    gx_stack = G.ravel(order="F")
    Hstack = H.transpose(0, 2, 1).reshape((d * N, d), order="F")

    # ── baselines for relative gaps ─────────────────────────────────────────
    x_avg0 = X.mean(axis=1)
    norm_x0, f_gap = None, None
    if "x_opt" in prm and prm["x_opt"] is not None and np.all(np.isfinite(prm["x_opt"])):
        norm_x0 = max(float(np.linalg.norm(x_avg0 - prm["x_opt"])), 1e-12)
    if "f_opt" in prm and prm["f_opt"] is not None and np.isfinite(prm["f_opt"]):
        f0 = float(np.mean([fi(x_avg0) for fi in f]))
        f_gap = max(abs(f0 - prm["f_opt"]), 1e-12)

    # ── bookkeeping ─────────────────────────────────────────────────────────
    fail_flag = False
    fail_reason = ""
    comm_total = 0.0
    t_start = time.perf_counter()

    Gtrue = G.copy()
    Htrue = H.copy()
    Htrue_klazy = H.copy()     # grad^2f at last Klazy update point
    _gtrue_valid = True

    # ═══════════════════════════ MAIN LOOP ══════════════════════════════════
    k = 0
    for k in range(K):
        if not np.all(np.isfinite(X)) or np.any(np.abs(X) > 1e10):
            fail_flag = True; fail_reason = "numerical overflow"; break

        if not _gtrue_valid:
            for i in range(N):
                gi, Hi = compute_info(X[:, i], prm, i, "both")
                Gtrue[:, i] = gi
                Htrue[:, :, i] = Hi
        _gtrue_valid = False

        # ── (A) Pre-mixing: x̃=W^tau x, g̃=W^tau g, H̃=W^tau H ─────────────────
        Wk = get_W(prm["W"], k)
        prm["NC"] = _log_consensus_rounds(prm, k, Wk)
        if prm.get("adaptive_ce", False):
            _ace_ratio = prm["NC"] / _ace_nc0
            prm["Klazy"] = max(1, int(_ace_klazy0 / max(1.0, _ace_ratio)))
            if prm.get("compressH", False):
                prm["compressParam"] = min(d**2, int(_ace_cp0 * max(1.0, _ace_ratio)))

        Xvec_pre = X.ravel(order="F")
        gx_pre = gx_stack.copy()
        Hstack_pre = Hstack.copy()
        _nc_mat = 0 if prm.get("skip_H_premix", False) else min(prm["NC"], prm.get("NC_mat_cap", prm["NC"]))
        for _t in range(prm["NC"]):
            Xvec_pre = _mix_vec(Wk, Xvec_pre, d, N)
            gx_pre = _mix_vec(Wk, gx_pre, d, N)
            if _t < _nc_mat:
                Hstack_pre = _mix_mat(Wk, Hstack_pre, d, N)
        X_tilde = Xvec_pre.reshape(d, N, order="F")
        G_tilde = gx_pre.reshape(d, N, order="F")
        H_tilde = Hstack_pre.reshape(d, N, d, order="F").transpose(0, 2, 1)

        if prm["countComm"]:
            comm_total += compute_comm_cost(d, Wk, "vector", rounds=prm["NC"]) * 2
            comm_total += compute_comm_cost(d, Wk, "full_matrix", rounds=_nc_mat)

        # ── (B) Regularised Newton step on pre-mixed quantities ──────────────
        H_sym = 0.5 * (H_tilde + H_tilde.transpose(1, 0, 2))     # (d,d,N)
        g_norms = np.linalg.norm(G_tilde, axis=0)                 # (N,)
        lam_vec = np.sqrt(prm["M"] * g_norms)                     # (N,)

        H_sym_b = H_sym.transpose(2, 0, 1)                        # (N,d,d)
        eig_min = np.linalg.eigvalsh(H_sym_b)[:, 0]               # (N,)
        delta_vec = np.maximum(0.0, -eig_min)                      # (N,)

        reg = (lam_vec + delta_vec)[:, None, None] * np.eye(d)
        H_reg_b = H_sym_b + reg                                   # (N,d,d)

        S = np.zeros((d, N))
        nonzero_grad = g_norms > 0
        if np.any(nonzero_grad):
            try:
                sol = np.linalg.solve(
                    H_reg_b[nonzero_grad],
                    G_tilde.T[nonzero_grad, :, None],
                )[:, :, 0]
                S[:, nonzero_grad] = -sol.T
            except np.linalg.LinAlgError:
                for i in np.where(nonzero_grad)[0]:
                    S[:, i] = -np.linalg.lstsq(
                        H_reg_b[i], G_tilde[:, i], rcond=None)[0]
        Y = X_tilde + S

        # ── (C) Post-mixing: x_{k+1} = W^t y ───────────────────────────────
        Yvec = Y.ravel(order="F")
        for _ in range(prm["NC"]):
            Yvec = _mix_vec(Wk, Yvec, d, N)
        X = Yvec.reshape(d, N, order="F")
        if prm["countComm"]:
            comm_total += compute_comm_cost(d, Wk, "vector", rounds=prm["NC"])

        # ── (D) Tracker updates ──────────────────────────────────────────────
        Gnext = np.zeros((d, N))
        Hnext = np.zeros((d, d, N))
        for i in range(N):
            gi, Hi = compute_info(X[:, i], prm, i, "both")
            Gnext[:, i] = gi
            Hnext[:, :, i] = Hi
        # g_{k+1} = W^t (g̃ + gradf(x_{k+1}) - gradf(x_k))
        gx_stack = gx_pre + (Gnext - Gtrue).ravel(order="F")
        for _ in range(prm["NC"]):
            gx_stack = _mix_vec(Wk, gx_stack, d, N)
        G = gx_stack.reshape(d, N, order="F")
        if prm["countComm"]:
            comm_total += compute_comm_cost(d, Wk, "vector", rounds=prm["NC"])

        Gtrue[:] = Gnext
        Htrue[:] = Hnext
        _gtrue_valid = True

        # H_{k+1} = sym(W^t (H̃ + grad^2f(x_{k+1}) - grad^2f(x_{last_klazy})))
        if (k + 1) % prm["Klazy"] == 0:
            Hdiff = Hnext - Htrue_klazy
            if prm["compressH"]:
                for i in range(N):
                    Hdiff[:, :, i] = compress(
                        Hdiff[:, :, i], prm["compressor"], prm["compressParam"])
            Hdiff_stack = Hdiff.transpose(0, 2, 1).reshape((d * N, d), order="F")
            Hstack = Hstack_pre + Hdiff_stack
            _nc_mat_klazy = min(prm["NC"], prm.get("NC_mat_cap", prm["NC"]))
            for _ in range(_nc_mat_klazy):
                Hstack = _mix_mat(Wk, Hstack, d, N)
            H_hat = Hstack.reshape(d, N, d, order="F").transpose(0, 2, 1)
            H = 0.5 * (H_hat + H_hat.transpose(1, 0, 2))
            Htrue_klazy[:] = Hnext
            if prm["countComm"]:
                comm_total += compute_comm_cost(
                    d, Wk, prm["compressor"], prm["compressParam"], _nc_mat_klazy)

        # ── logging ──────────────────────────────────────────────────────────
        x_avg = X.mean(axis=1)
        log_data["gradNrm"][k] = float(np.linalg.norm(G.mean(axis=1)))
        log_data["cons"][k] = float(np.linalg.norm(X - x_avg[:, None], "fro") / np.sqrt(N))
        log_data["combo"][k] = log_data["gradNrm"][k] + log_data["cons"][k]
        log_data["ValueF"][k] = float(np.mean([fi(x_avg) for fi in f]))
        log_data["xBar"][k] = x_avg
        log_data["Mavg"][k] = float(prm["M"])
        log_data["commCost"][k] = comm_total / 1024 ** 2
        log_data["timeCost"][k] = time.perf_counter() - t_start
        log_data["NC"][k] = float(prm["NC"])

        if norm_x0 is not None:
            log_data["relX"][k] = float(np.linalg.norm(x_avg - prm["x_opt"])) / norm_x0
        if f_gap is not None:
            log_data["relF"][k] = abs(log_data["ValueF"][k] - prm["f_opt"]) / f_gap
        if k > 0 and not np.isnan(log_data["gradNrm"][k - 1]):
            log_data["rk"][k] = log_data["gradNrm"][k] / max(1e-12, log_data["gradNrm"][k - 1])

        if prm["verbose"] and (k + 1) % print_freq == 0:
            print_progress(k + 1, log_data["gradNrm"][k], log_data["cons"][k],
                           log_data["ValueF"][k],
                           relX=log_data["relX"][k], relF=log_data["relF"][k],
                           commCost=log_data["commCost"][k])

        if prm["tolType"] == "combo" and log_data["combo"][k] < prm["tol"]:
            break
        if (prm["tolType"] == "relX"
                and not np.isnan(log_data["relX"][k])
                and log_data["relX"][k] < prm["tol"]):
            break
        if (prm["tolType"] == "relF"
                and not np.isnan(log_data["relF"][k])
                and log_data["relF"][k] < prm["tol"]):
            break
        _tol_relF = prm.get("tol_relF", 1e-12)
        if _tol_relF > 0 and not np.isnan(log_data["relF"][k]) and log_data["relF"][k] < _tol_relF:
            break

    # ── finalise ─────────────────────────────────────────────────────────────
    log_data = trim_log(log_data, k + 1)
    log_data["fail"] = fail_flag
    log_data["failReason"] = fail_reason
    x_all = X.ravel(order="F")
    return x_all, log_data
