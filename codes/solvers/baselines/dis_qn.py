"""
dis_qn.py - DisQN: Distributed Quasi-Newton with local BFGS.
Ported from MATLAB: DisQN.m
"""

from __future__ import annotations
import time
import numpy as np
from utils.helper.log_utils import init_log, trim_log, compute_comm_cost, print_progress
from utils.helper.eval_utils import compute_info, eval_all
from utils.helper.graph import get_W


def dis_qn(x0: np.ndarray, prm: dict):
    """
    DisQN: Distributed Quasi-Newton with gradient tracking and BFGS.
    """
    p = {
        "alpha": 1e-2, "decay_alpha": False,
        "maxIt": 200, "tol": 1e-10, "tolType": "combo",
        "verbose": True, "countComm": True, "memoryLimitMB": np.inf,
        "info": 0,
        "x_opt": None, "f_opt": None,
    }
    p.update(prm)
    prm = p

    N = prm["Nagent"]; d = prm["dim"]; f = prm["f"]
    K = prm["maxIt"]
    print_freq = max(1, K // 25)

    X = np.tile(x0.ravel()[:, None], (1, N))
    V = np.zeros((d, N))   # gradient tracker
    C = np.tile(np.eye(d)[:, :, None], (1, 1, N))   # inverse-Hessian approx
    D = np.zeros((d, N))   # local search direction
    Z = np.zeros((d, N))   # mixed direction

    for i in range(N):
        gi, _ = compute_info(X[:, i], prm, i, "grad")
        V[:, i] = gi
        D[:, i] = -C[:, :, i] @ V[:, i]

    Wk0 = get_W(prm["W"], 0)
    Z = Wk0 @ D.T           # (N, d) → then transpose
    Z = Z.T                  # (d, N)

    log_data = init_log(K)
    log_data["xBar"] = np.full((K, d), np.nan)

    denom_x0, denom_f0 = None, None
    if prm["x_opt"] is not None and np.all(np.isfinite(prm["x_opt"])):
        denom_x0 = max(np.linalg.norm(x0.ravel() - prm["x_opt"]), 1e-12)
    if prm["f_opt"] is not None and np.isfinite(prm["f_opt"]):
        f0v = float(np.mean([fi(x0.ravel()) for fi in f]))
        denom_f0 = max(abs(f0v - prm["f_opt"]), 1e-12)

    fail_flag = False; fail_reason = ""
    comm_total = 0.0; t_start = time.perf_counter()

    k = 0
    for k in range(K):
        if not np.all(np.isfinite(X)) or np.any(np.abs(X) > 1e10):
            fail_flag = True; fail_reason = "numerical"; break

        alpha_k = prm["alpha"] / np.sqrt(k + 1) if prm["decay_alpha"] else prm["alpha"]
        Wk = get_W(prm["W"], k)

        # (1) primal update via consensus + direction
        X_prev = X.copy()
        X_new = np.zeros((d, N))
        for i in range(N):
            xi_new = np.zeros(d)
            for j in range(N):
                xi_new += Wk[i, j] * (X[:, j] + alpha_k * Z[:, j])
            X_new[:, i] = xi_new
        X = X_new

        # (2) gradient tracking - batch-compute grads first to avoid O(N^2) calls
        V_prev = V.copy()
        G_new = np.zeros((d, N))
        G_old = np.zeros((d, N))
        for j in range(N):
            G_new[:, j], _ = compute_info(X[:, j], prm, j, "grad")
            G_old[:, j], _ = compute_info(X_prev[:, j], prm, j, "grad")
        # V_new[:,i] = sum_j W[i,j] * (V[:,j] + G_new[:,j] - G_old[:,j])
        delta_G = G_new - G_old                       # (d, N)
        V_new = (V + delta_G) @ Wk.T                  # (d, N)
        V = V_new

        # (3) BFGS update - use gradient-TRACKING secant (intentional design):
        # V converges to the global average gradient, so V_new - V_old approximates
        # H_bar @ s_bar, building C_i → H_bar^{-1} (global curvature, not local H_i^{-1}).
        # This is preferable because the search direction D = -C @ V targets the global optimum.
        for i in range(N):
            s = X[:, i] - X_prev[:, i]
            y = V[:, i] - V_prev[:, i]    # tracking-gradient secant → global Hessian approx
            ys = y @ s
            ys_thresh = 1e-8 * (np.linalg.norm(s) * np.linalg.norm(y))
            if ys > max(ys_thresh, 1e-16):
                Id = np.eye(d)
                C[:, :, i] = ((Id - np.outer(s, y) / ys) @ C[:, :, i]
                              @ (Id - np.outer(y, s) / ys)
                              + np.outer(s, s) / ys)
            D[:, i] = -C[:, :, i] @ V[:, i]

        # (4) propagate direction
        Z = (Wk @ D.T).T

        # Communication accounting:
        #   Round 1 - primal mixing in step (1): X + alpha*Z mixed together
        #   Round 2 - gradient tracking in step (2): V updated via W-consensus
        #   Round 3 - direction propagation in step (4): Z = W D
        # Each round exchanges one d-dimensional vector per edge.
        if prm["countComm"]:
            comm_total += 3 * compute_comm_cost(d, Wk, "vector")

        x_bar = X.mean(axis=1)
        # Evaluate global objective at consensus average (consistent with other algorithms)
        fval = float(np.mean([fi(x_bar) for fi in f]))
        log_data["gradNrm"][k] = float(np.linalg.norm(V.mean(axis=1)))
        log_data["cons"][k] = float(np.sqrt(np.sum((X - x_bar[:, None]) ** 2) / N))
        log_data["combo"][k] = log_data["gradNrm"][k] + log_data["cons"][k]
        log_data["ValueF"][k] = fval
        log_data["xBar"][k] = x_bar
        log_data["commCost"][k] = comm_total / 1024 ** 2
        log_data["timeCost"][k] = time.perf_counter() - t_start
        if denom_x0 is not None:
            log_data["relX"][k] = np.linalg.norm(x_bar - prm["x_opt"]) / denom_x0
        if denom_f0 is not None:
            log_data["relF"][k] = abs(fval - prm["f_opt"]) / denom_f0
        if k > 0 and not np.isnan(log_data["gradNrm"][k - 1]):
            log_data["rk"][k] = log_data["gradNrm"][k] / max(1e-12, log_data["gradNrm"][k - 1])

        if prm["verbose"] and (k + 1) % print_freq == 0:
            print_progress(k + 1, log_data["gradNrm"][k], log_data["cons"][k],
                           fval, relX=log_data["relX"][k], relF=log_data["relF"][k])

        if prm["tolType"] == "combo" and log_data["combo"][k] < prm["tol"]:
            break
        if prm["tolType"] == "relX" and denom_x0 is not None and log_data["relX"][k] < prm["tol"]:
            break
        if prm["tolType"] == "relF" and denom_f0 is not None and log_data["relF"][k] < prm["tol"]:
            break

    log_data = trim_log(log_data, k + 1)
    log_data["fail"] = fail_flag; log_data["failReason"] = fail_reason
    return X.ravel(order="F"), log_data
