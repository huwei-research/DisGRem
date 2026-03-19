"""
ce_ada_disgrem.py – CeAdaDisGrem: Adaptive version with per-agent M_i.

Aligned with Algorithm 1 + AdaDisGrem adaptive rule from the paper:
  (A) Pre-mixing, (B) Newton (sym + eigval δ), (C) Post-mixing, (D) Trackers.
  M_i update uses secant Hessian-Lipschitz estimate:
    M̂_k ← max(M̂_{k−1}, ‖∇²f(x_{k+1})−∇²f(x_k)‖ / ‖x_{k+1}−x_k‖)
  with practical γ-decay and ζ-scaling.
"""

from __future__ import annotations
import time
import numpy as np
from utils.helper.log_utils import init_log, trim_log, compute_comm_cost, print_progress
from utils.helper.eval_utils import compute_info
from utils.helper.compress import compress
from utils.helper.graph import get_W


def _mix_vec(W, v, d, N):
    """(W ⊗ I_d) @ v  without forming the Kronecker product.  O(dN²)."""
    return (v.reshape(d, N, order='F') @ W.T).ravel(order='F')


def _mix_mat(W, M, d, N):
    """(W ⊗ I_d) @ M  for M of shape (dN, cols).  O(dN² · cols)."""
    cols = M.shape[1]
    M3 = M.reshape(d, N, cols, order='F')
    return (M3.transpose(0, 2, 1) @ W.T).transpose(0, 2, 1).reshape(
        d * N, cols, order='F')


def ce_ada_disgrem(x0: np.ndarray, prm: dict):
    """
    CeAdaDisGrem: Adaptive Decentralised Newton with per-agent M_i update.

    Extra prm fields beyond CeDisGrem:
        .gamma  (0 < γ < 1) forgetting factor  [0.8]
        .zeta   (> 0)       scale on deltas     [1.0]
        .contr  contraction factor              [10]
    """
    p = {
        "maxIt": 200, "tol": 1e-16, "tolType": "combo",
        "verbose": True, "M": 10.0, "gamma": 0.5, "zeta": 1.0, "contr": 3.0,
        "compressH": True, "compressor": "topk", "compressParam": None,
        "NC": 3, "Klazy": None, "countComm": True,
        "memoryLimitMB": np.inf, "info": 0,
    }
    p.update(prm)
    prm = p

    N = prm["Nagent"]; d = prm["dim"]; f = prm["f"]
    K = prm["maxIt"]
    print_freq = max(1, K // 25)

    assert N == len(f), f"Nagent={N} but len(f)={len(f)}"
    assert d == len(x0.ravel()), f"dim={d} but x0 has length {len(x0.ravel())}"
    assert prm.get("W") is not None, "Mixing matrix W is required"
    assert prm["M"] > 0, f"Regularisation M must be positive, got {prm['M']}"

    if prm["compressParam"] is None:
        prm["compressParam"] = max(1, min(round(0.10 * d ** 2), int(1e5)))
    if prm["Klazy"] is None:
        prm["Klazy"] = max(1, min(K // 4, 80))

    X = np.tile(x0.ravel()[:, None], (1, N))
    G = np.zeros((d, N))
    H = np.zeros((d, d, N))
    gx_stack = np.zeros(d * N)
    Hstack = np.zeros((d * N, d))
    M_vec = np.full(N, prm["M"])   # per-agent M_i

    log_data = init_log(K)
    log_data["xBar"] = np.full((K, d), np.nan)

    for i in range(N):
        gi, Hi = compute_info(X[:, i], prm, i, "both")
        G[:, i] = gi
        H[:, :, i] = Hi

    gx_stack = G.ravel(order="F")
    Hstack = H.transpose(0, 2, 1).reshape((d * N, d), order="F")

    x_avg0 = X.mean(axis=1)
    norm_x0, f_gap = None, None
    if "x_opt" in prm and prm["x_opt"] is not None and np.all(np.isfinite(prm["x_opt"])):
        norm_x0 = max(np.linalg.norm(x_avg0 - prm["x_opt"]), 1e-12)
    if "f_opt" in prm and prm["f_opt"] is not None and np.isfinite(prm["f_opt"]):
        f0 = float(np.mean([fi(x_avg0) for fi in f]))
        f_gap = max(abs(f0 - prm["f_opt"]), 1e-12)

    fail_flag = False; fail_reason = ""
    comm_total = 0.0; t_start = time.perf_counter()

    Gtrue = G.copy()
    Htrue = H.copy()
    Htrue_klazy = H.copy()     # ∇²f at last Klazy update point
    _gtrue_valid = True

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

        X_old = X.copy()

        # ── (A) Pre-mixing: x̃=W^τ x, g̃=W^τ g, H̃=W^τ H ─────────────────
        Wk = get_W(prm["W"], k)

        Xvec_pre = X.ravel(order="F")
        gx_pre = gx_stack.copy()
        Hstack_pre = Hstack.copy()
        for _ in range(prm["NC"]):
            Xvec_pre = _mix_vec(Wk, Xvec_pre, d, N)
            gx_pre = _mix_vec(Wk, gx_pre, d, N)
            Hstack_pre = _mix_mat(Wk, Hstack_pre, d, N)
        X_tilde = Xvec_pre.reshape(d, N, order="F")
        G_tilde = gx_pre.reshape(d, N, order="F")
        H_tilde = Hstack_pre.reshape(d, N, d, order="F").transpose(0, 2, 1)

        if prm["countComm"]:
            comm_total += compute_comm_cost(d, Wk, "vector", rounds=prm["NC"]) * 2
            comm_total += compute_comm_cost(d, Wk, "full_matrix", rounds=prm["NC"])

        # ── (B) Adaptive regularised Newton step ─────────────────────────────
        H_sym = 0.5 * (H_tilde + H_tilde.transpose(1, 0, 2))     # (d,d,N)
        g_norms = np.linalg.norm(G_tilde, axis=0)                 # (N,)
        lam_vec = np.sqrt(M_vec * g_norms)                        # (N,)

        H_sym_b = H_sym.transpose(2, 0, 1)                        # (N,d,d)
        eig_min = np.linalg.eigvalsh(H_sym_b)[:, 0]               # (N,)
        delta_vec = np.maximum(0.0, -eig_min)                      # (N,)

        reg = (lam_vec + delta_vec)[:, None, None] * np.eye(d)
        H_reg_b = H_sym_b + reg                                   # (N,d,d)

        try:
            S = -np.linalg.solve(H_reg_b, G_tilde.T[:, :, None])[:, :, 0].T
        except np.linalg.LinAlgError:
            S = np.zeros((d, N))
            for i in range(N):
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
        Htrue_old = Htrue.copy()

        # g_{k+1} = W^t (g̃ + ∇f(x_{k+1}) − ∇f(x_k))
        gx_stack = gx_pre + (Gnext - Gtrue).ravel(order="F")
        for _ in range(prm["NC"]):
            gx_stack = _mix_vec(Wk, gx_stack, d, N)
        G = gx_stack.reshape(d, N, order="F")
        if prm["countComm"]:
            comm_total += compute_comm_cost(d, Wk, "vector", rounds=prm["NC"])

        Gtrue[:] = Gnext
        Htrue[:] = Hnext
        _gtrue_valid = True

        # H_{k+1} = sym(W^t (H̃ + ∇²f(x_{k+1}) − ∇²f(x_{last_klazy})))
        if (k + 1) % prm["Klazy"] == 0:
            Hdiff = Hnext - Htrue_klazy
            if prm["compressH"]:
                for i in range(N):
                    Hdiff[:, :, i] = compress(
                        Hdiff[:, :, i], prm["compressor"], prm["compressParam"])
            Hdiff_stack = Hdiff.transpose(0, 2, 1).reshape((d * N, d), order="F")
            Hstack = Hstack_pre + Hdiff_stack
            for _ in range(prm["NC"]):
                Hstack = _mix_mat(Wk, Hstack, d, N)
            H_hat = Hstack.reshape(d, N, d, order="F").transpose(0, 2, 1)
            H = 0.5 * (H_hat + H_hat.transpose(1, 0, 2))
            Htrue_klazy[:] = Hnext
            if prm["countComm"]:
                comm_total += compute_comm_cost(
                    d, Wk, prm["compressor"], prm["compressParam"], prm["NC"])

        # ── M_i update: secant Hessian-Lipschitz estimate ────────────────────
        for i in range(N):
            s_i = X[:, i] - X_old[:, i]
            ns = np.linalg.norm(s_i)
            if ns > 1e-16:
                delta_hl = np.linalg.norm(
                    Hnext[:, :, i] - Htrue_old[:, :, i], "fro") / ns
                M_vec[i] = max(
                    prm["zeta"] * min(delta_hl, prm["contr"] * prm["M"]),
                    prm["gamma"] * M_vec[i])

        # ── logging ──────────────────────────────────────────────────────────
        x_avg = X.mean(axis=1)
        log_data["gradNrm"][k] = float(np.linalg.norm(G.mean(axis=1)))
        log_data["cons"][k] = float(np.linalg.norm(X - x_avg[:, None], "fro") / np.sqrt(N))
        log_data["combo"][k] = log_data["gradNrm"][k] + log_data["cons"][k]
        log_data["ValueF"][k] = float(np.mean([fi(x_avg) for fi in f]))
        log_data["xBar"][k] = x_avg
        log_data["Mavg"][k] = float(M_vec.mean())
        log_data["commCost"][k] = comm_total / 1024 ** 2
        log_data["timeCost"][k] = time.perf_counter() - t_start

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
                           Mavg=log_data["Mavg"][k])

        if prm["tolType"] == "combo" and log_data["combo"][k] < prm["tol"]:
            break
        if prm["tolType"] == "relX" and not np.isnan(log_data["relX"][k]) and log_data["relX"][k] < prm["tol"]:
            break
        if prm["tolType"] == "relF" and not np.isnan(log_data["relF"][k]) and log_data["relF"][k] < prm["tol"]:
            break

    log_data = trim_log(log_data, k + 1)
    log_data["fail"] = fail_flag
    log_data["failReason"] = fail_reason
    return X.ravel(order="F"), log_data
