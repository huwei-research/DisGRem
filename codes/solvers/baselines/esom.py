"""
esom.py – ESOM: Exact Second-Order Method (Mokhtari et al.).
Ported from MATLAB: ESOM.m
"""

from __future__ import annotations
import time
import numpy as np
from utils.helper.log_utils import init_log, trim_log, compute_comm_cost, print_progress
from utils.helper.eval_utils import compute_info
from utils.helper.graph import get_W


def esom(x0: np.ndarray, prm: dict):
    """
    ESOM algorithm.

    Extra prm fields:
        .alpha     penalty parameter     [1.0]
        .epsl      regularisation        [1e-6]
        .chi       inner refinement iters [3]
    """
    p = {
        "esom_penalty": 1.0,   # ADMM penalty ρ — do NOT confuse with gradient step α
        # Paper (Mokhtari et al. 2016, dynamic ESOM) defines D_ii = H_i + 2ρ(1-w_ii)I + I.
        # The "+1" unit regularisation (esom_unit_reg) is NOT a small ε but a structural term
        # from the ADMM augmented Lagrangian.  Using only epsl≈0 (as before) caused catastrophic
        # divergence on near-flat objectives (e.g. linlog) because D became near-singular.
        "esom_unit_reg": 1.0,  # the "+I" term from the paper: D = H + 2ρ(1-w_ii)I + unit_reg*I
        "epsl": 1e-6, "chi": 3,
        "maxIt": 200, "tol": 1e-8, "tolType": "combo",
        "verbose": True, "countComm": True, "memoryLimitMB": np.inf,
        "info": 0, "NC": 1,
        "x_opt": None, "f_opt": None,
    }
    p.update(prm)
    prm = p

    N = prm["Nagent"]; d = prm["dim"]; f = prm["f"]
    K = prm["maxIt"]; chi = prm["chi"]
    # Use algorithm-specific penalty; never override with the gradient step size 'alpha'
    alpha = prm.get("esom_penalty", 1.0)
    unit_reg = prm.get("esom_unit_reg", 1.0)
    epsl = prm["epsl"]
    print_freq = max(1, K // 25)

    X = np.tile(x0.ravel()[:, None], (1, N))
    Q = np.zeros((d, N))   # dual variables

    log_data = init_log(K)
    log_data["xBar"] = np.full((K, d), np.nan)

    gap_x0, gap_f0 = None, None
    if prm["x_opt"] is not None and np.all(np.isfinite(prm["x_opt"])):
        gap_x0 = max(np.linalg.norm(x0.ravel() - prm["x_opt"]), 1e-12)
    if prm["f_opt"] is not None and np.isfinite(prm["f_opt"]):
        f0 = float(np.mean([fi(x0.ravel()) for fi in f]))
        gap_f0 = max(abs(f0 - prm["f_opt"]), 1e-12)

    fail_flag = False; fail_reason = ""
    comm_total = 0.0; t_start = time.perf_counter()

    k = 0
    for k in range(K):
        if not np.all(np.isfinite(X)) or np.any(np.abs(X) > 1e10):
            fail_flag = True; fail_reason = "numerical"; break

        Wk = get_W(prm["W"], k)
        Mix = np.kron(Wk, np.eye(d))

        # (1) pre-mixing
        Xvec = X.ravel(order="F")
        for _ in range(prm["NC"]):
            Xvec = Mix @ Xvec
        X = Xvec.reshape(d, N, order="F")
        if prm["countComm"]:
            comm_total += compute_comm_cost(d, Wk, "vector", rounds=prm["NC"])

        # (2) local grad & Hessian
        G_loc = np.zeros((d, N)); H_loc = np.zeros((d, d, N))
        for i in range(N):
            gi, Hi = compute_info(X[:, i], prm, i, "both")
            G_loc[:, i] = gi; H_loc[:, :, i] = Hi

        # (3) consensus term x̄_j
        Xvec2 = X.ravel(order="F")
        for _ in range(prm["NC"]):
            Xvec2 = Mix @ Xvec2
        X_nbr = Xvec2.reshape(d, N, order="F")
        if prm["countComm"]:
            comm_total += compute_comm_cost(d, Wk, "vector", rounds=prm["NC"])

        # (4) modified gradient ḡ_i
        g_bar = np.zeros((d, N))
        for i in range(N):
            w_ii = Wk[i, i]
            nbr_sum = X_nbr[:, i] - w_ii * X[:, i]
            g_bar[:, i] = (G_loc[:, i] + Q[:, i]
                           + alpha * (1 - w_ii) * X[:, i]
                           - alpha * nbr_sum)

        # (5) D_inv blocks
        # Paper: D_ii = H_i + 2ρ(1-w_ii)I + I  (the "+I" = unit_reg is structural, not ε-small)
        D_inv = np.zeros((d, d, N))
        for i in range(N):
            w_ii = Wk[i, i]
            D = H_loc[:, :, i] + (2 * alpha * (1 - w_ii) + unit_reg + epsl) * np.eye(d)
            try:
                D_inv[:, :, i] = np.linalg.inv(D)
            except np.linalg.LinAlgError:
                D_inv[:, :, i] = np.linalg.pinv(D)

        # (6) ESOM-0 direction
        dirs = np.zeros((d, N, chi + 1))
        for i in range(N):
            dirs[:, i, 0] = -D_inv[:, :, i] @ g_bar[:, i]

        # (7) inner refinement — each round requires neighbours to share dirs
        # B_ij: diagonal B_ii = ρ(1-w_ii), off-diagonal B_ij = ρ*w_ij
        # Self term B_ii*d_i^(k) is ALWAYS included regardless of w_ii > 0.
        # Bug fix: previous code skipped self contribution when w_ii=0.
        for kk in range(chi):
            if prm["countComm"]:
                comm_total += compute_comm_cost(d, Wk, "vector", rounds=prm["NC"])
            for i in range(N):
                # Self contribution: B_ii = alpha*(1-w_ii) — always include
                tmp = alpha * (1.0 - Wk[i, i]) * dirs[:, i, kk]
                # Neighbor contributions: B_ij = alpha*w_ij for j != i
                for j in range(N):
                    if i != j and Wk[i, j] > 0:
                        tmp += alpha * Wk[i, j] * dirs[:, j, kk]
                dirs[:, i, kk + 1] = D_inv[:, :, i] @ (tmp - g_bar[:, i])

        d_K = dirs[:, :, -1]

        # (8) primal update
        X = X + d_K

        # (9) post-mixing
        Xvec = X.ravel(order="F")
        for _ in range(prm["NC"]):
            Xvec = Mix @ Xvec
        X = Xvec.reshape(d, N, order="F")
        if prm["countComm"]:
            comm_total += compute_comm_cost(d, Wk, "vector", rounds=prm["NC"])

        # (10) dual update
        Xvec3 = X.ravel(order="F")
        for _ in range(prm["NC"]):
            Xvec3 = Mix @ Xvec3
        X_nbr_new = Xvec3.reshape(d, N, order="F")
        for i in range(N):
            w_ii = Wk[i, i]
            nbr_sum_new = X_nbr_new[:, i] - w_ii * X[:, i]
            Q[:, i] += alpha * (1 - w_ii) * X[:, i] - alpha * nbr_sum_new

        # logging
        x_bar = X.mean(axis=1)
        g_avg = G_loc.mean(axis=1)
        log_data["gradNrm"][k] = float(np.linalg.norm(g_avg))
        log_data["cons"][k] = float(np.linalg.norm(X - x_bar[:, None], "fro") / np.sqrt(N))
        log_data["combo"][k] = log_data["gradNrm"][k] + log_data["cons"][k]
        log_data["ValueF"][k] = float(np.mean([fi(x_bar) for fi in f]))
        log_data["xBar"][k] = x_bar
        log_data["commCost"][k] = comm_total / 1024 ** 2
        log_data["timeCost"][k] = time.perf_counter() - t_start
        if gap_x0 is not None:
            log_data["relX"][k] = np.linalg.norm(x_bar - prm["x_opt"]) / gap_x0
        if gap_f0 is not None:
            log_data["relF"][k] = abs(log_data["ValueF"][k] - prm["f_opt"]) / gap_f0
        if k > 0 and not np.isnan(log_data["gradNrm"][k - 1]):
            log_data["rk"][k] = log_data["gradNrm"][k] / max(1e-12, log_data["gradNrm"][k - 1])

        if prm["verbose"] and (k + 1) % print_freq == 0:
            print_progress(k + 1, log_data["gradNrm"][k], log_data["cons"][k],
                           log_data["ValueF"][k],
                           relX=log_data["relX"][k], relF=log_data["relF"][k])

        if prm["tolType"] == "combo" and log_data["combo"][k] < prm["tol"]:
            break
        if prm["tolType"] == "relX" and gap_x0 is not None and log_data["relX"][k] < prm["tol"]:
            break
        if prm["tolType"] == "relF" and gap_f0 is not None and log_data["relF"][k] < prm["tol"]:
            break

    log_data = trim_log(log_data, k + 1)
    log_data["fail"] = fail_flag; log_data["failReason"] = fail_reason
    return X.ravel(order="F"), log_data
