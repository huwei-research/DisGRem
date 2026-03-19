"""
dis_greqm.py – DisGreQm: Distributed Gradient-Regularised Quasi-Newton (BFGS).
Ported from MATLAB: DisGreQm.m
"""

from __future__ import annotations
import time
import numpy as np
from utils.helper.log_utils import init_log, trim_log, compute_comm_cost, print_progress
from utils.helper.eval_utils import compute_info
from utils.helper.graph import get_W


def dis_greqm(x0: np.ndarray, prm: dict):
    """
    DisGreQm: Distributed Gradient-Regularised Quasi-Newton (full-matrix BFGS).

    Uses gradient tracking + BFGS Hessian approximation.
    """
    p = {
        "maxIt": 200, "tol": 1e-16, "tolType": "combo",
        "verbose": True, "M": 10.0, "info": 0,
        "compressH": True, "compressor": "topk", "compressParam": None,
        "NC": 1, "Klazy": 5, "countComm": True,
        "memoryLimitMB": np.inf,
    }
    p.update(prm)
    prm = p

    N = prm["Nagent"]; d = prm["dim"]; f = prm["f"]
    K = prm["maxIt"]
    print_freq = max(1, K // 25)

    if prm["compressParam"] is None:
        prm["compressParam"] = max(1, min(round(0.05 * d ** 2), int(1e5)))

    X = np.tile(x0.ravel()[:, None], (1, N))
    G = np.zeros((d, N))
    G_true = np.zeros((d, N))    # true local gradients (for BFGS secant)
    B = np.tile(np.eye(d)[:, :, None], (1, 1, N))        # (d, d, N) BFGS approx
    B_prev = B.copy()
    g_stack = np.zeros(d * N)

    log_data = init_log(K)
    log_data["xBar"] = np.full((K, d), np.nan)

    x_bar0 = X.mean(axis=1)
    gap_x0, gap_f0 = None, None
    if "x_opt" in prm and prm["x_opt"] is not None and np.all(np.isfinite(prm["x_opt"])):
        gap_x0 = max(np.linalg.norm(x_bar0 - prm["x_opt"]), 1e-12)
    if "f_opt" in prm and prm["f_opt"] is not None and np.isfinite(prm["f_opt"]):
        f0 = float(np.mean([fi(x_bar0) for fi in f]))
        gap_f0 = max(abs(f0 - prm["f_opt"]), 1e-12)

    for i in range(N):
        gi, _ = compute_info(X[:, i], prm, i, "grad")
        G[:, i] = gi
        G_true[:, i] = gi    # initialise true-gradient cache
    g_stack = G.ravel(order="F")

    fail_flag = False; fail_reason = ""
    comm_total = 0.0; t_start = time.perf_counter()

    k = 0
    for k in range(K):
        if not np.all(np.isfinite(X)) or np.any(np.abs(X) > 1e10):
            fail_flag = True; fail_reason = "numerical"; break

        X_old = X.copy()
        G_old = G.copy()                    # tracking variable (for tracking update)
        G_true_prev = G_true.copy()         # true gradients at X_old (for BFGS secant)

        # regularised step
        S_step = np.zeros((d, N))
        for i in range(N):
            gi = G[:, i]; Bi = B[:, :, i]
            lam = np.sqrt(prm["M"] * np.linalg.norm(gi))
            e_diff = np.linalg.norm(Bi - B_prev[:, :, i], "fro")
            H_reg = Bi + (lam + e_diff) * np.eye(d)
            try:
                S_step[:, i] = -np.linalg.solve(H_reg, gi)
            except np.linalg.LinAlgError:
                S_step[:, i] = -np.linalg.lstsq(H_reg, gi, rcond=None)[0]

        X = X + S_step

        # consensus
        Wk = get_W(prm["W"], k)
        Mix = np.kron(Wk, np.eye(d))
        Xvec = X.ravel(order="F")
        for _ in range(prm["NC"]):
            Xvec = Mix @ Xvec
        X = Xvec.reshape(d, N, order="F")
        if prm["countComm"]:
            comm_total += compute_comm_cost(d, Wk, "vector", rounds=prm["NC"])

        # gradient tracking
        G_new = np.zeros((d, N))
        for i in range(N):
            G_new[:, i], _ = compute_info(X[:, i], prm, i, "grad")
        g_stack = g_stack + (G_new - G_old).ravel(order="F")
        for _ in range(prm["NC"]):
            g_stack = Mix @ g_stack
        G = g_stack.reshape(d, N, order="F")
        if prm["countComm"]:
            comm_total += compute_comm_cost(d, Wk, "vector", rounds=prm["NC"])
        G_true = G_new.copy()              # cache true gradients for next iteration

        # BFGS update: use TRUE gradient secant (G_new at x_new, G_true_prev at x_old)
        # then mix s and y across agents for distributed curvature sharing
        S_diff = X - X_old
        Y_diff = G_new - G_true_prev       # correct secant: ∇f(x_new) - ∇f(x_old)
        Svec = S_diff.ravel(order="F"); Yvec = Y_diff.ravel(order="F")
        for _ in range(prm["NC"]):
            Svec = Mix @ Svec; Yvec = Mix @ Yvec
        S_diff = Svec.reshape(d, N, order="F")
        Y_diff = Yvec.reshape(d, N, order="F")
        if prm["countComm"]:
            comm_total += 2 * compute_comm_cost(d, Wk, "vector", rounds=prm["NC"])

        B_prev = B.copy()
        for i in range(N):
            si = S_diff[:, i]; yi = Y_diff[:, i]
            # Relative curvature condition (Wolfe-style): s'y > eps * ||s|| * ||y||
            # Tighter than absolute threshold — avoids ill-conditioned BFGS updates
            ys_thresh = 1e-8 * (np.linalg.norm(si) * np.linalg.norm(yi))
            if si @ yi > max(ys_thresh, 1e-16):
                rho_i = 1.0 / (yi @ si)
                Bi = B[:, :, i]
                Id = np.eye(d)
                Bi = ((Id - rho_i * np.outer(si, yi)) @ Bi @ (Id - rho_i * np.outer(yi, si))
                      + rho_i * np.outer(yi, yi))
                # clip eigenvalues for stability
                eigvals, eigvecs = np.linalg.eigh((Bi + Bi.T) / 2)
                eigvals = np.clip(eigvals, 1e-6, 1e6)
                B[:, :, i] = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # logging
        x_bar = X.mean(axis=1)
        log_data["gradNrm"][k] = float(np.linalg.norm(G.mean(axis=1)))
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
