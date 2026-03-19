"""
diging.py – DIGing: Distributed Inexact Gradient and Gradient Tracking.

Reference:
    Nedic, A., Olshevsky, A., & Shi, W. (2017).
    Achieving geometric convergence for distributed optimization over
    time-varying graphs. SIAM Journal on Optimization, 27(4), 2597-2633.

Iteration:
    y^0        = ∇F(x^0)   (gradient tracker)
    x^{k+1}   = W x^k − α y^k
    y^{k+1}   = W y^k + ∇F(x^{k+1}) − ∇F(x^k)

DIGing mixes x first then applies gradient — following the original Nedic et al. formulation.
Converges linearly to exact consensus for strongly convex, smooth objectives.
"""

from __future__ import annotations
import time
import numpy as np
from utils.helper.log_utils import init_log, trim_log, compute_comm_cost, print_progress
from utils.helper.eval_utils import approx_grad
from utils.helper.graph import get_W


def diging(x0: np.ndarray, prm: dict):
    """
    DIGing algorithm.

    prm extra fields:
        .alpha        stepsize (required)
        .decay_alpha  bool – use α/√k  [False]
        .NC           consensus rounds  [1]
    """
    p = {
        "maxIt": 200, "tol": 1e-8, "tolType": "combo",
        "verbose": True, "alpha": None, "decay_alpha": False,
        "countComm": True, "memoryLimitMB": np.inf, "NC": 1,
        "x_opt": None, "f_opt": None,
        # DIGing-specific override: if 'alpha_diging' is set it takes priority.
        "alpha_diging": None,
    }
    p.update(prm)
    prm = p
    assert prm["alpha"] is not None, "stepsize alpha is required for DIGing"
    # DIGing uses alpha ≈ 0.5/L for best convergence.
    # Use algorithm-specific override if provided, otherwise 2.5× the base alpha
    # (corresponding to 0.5/L when base alpha = 0.2/L).
    if prm["alpha_diging"] is not None:
        prm["alpha"] = float(prm["alpha_diging"])
    else:
        prm["alpha"] = prm["alpha"] * 2.5   # scale: 0.2/L → 0.5/L

    N = prm["Nagent"]; d = prm["dim"]; f = prm["f"]
    K = prm["maxIt"]
    print_freq = max(1, K // 25)

    X = np.tile(x0.ravel()[:, None], (1, N))   # (d, N)
    G = np.zeros((d, N))
    for i in range(N):
        G[:, i] = approx_grad(f[i], x0.ravel())
    Y = G.copy()    # gradient tracker, initialised at ∇F(x^0)

    log_data = init_log(K)
    log_data["xBar"] = np.full((K, d), np.nan)

    x_avg0 = X.mean(axis=1)
    norm_x0, f_gap = None, None
    if prm["x_opt"] is not None and np.all(np.isfinite(prm["x_opt"])):
        norm_x0 = max(np.linalg.norm(x_avg0 - prm["x_opt"]), 1e-12)
    if prm["f_opt"] is not None and np.isfinite(prm["f_opt"]):
        f0 = float(np.mean([fi(x_avg0) for fi in f]))
        f_gap = max(abs(f0 - prm["f_opt"]), 1e-12)

    fail_flag = False; fail_reason = ""
    comm_total = 0.0; t_start = time.perf_counter()

    k = 0
    for k in range(K):
        if not np.all(np.isfinite(X)) or np.any(np.abs(X) > 1e12):
            fail_flag = True; fail_reason = "numerical overflow"; break

        # ── logging ────────────────────────────────────────────────────────────
        x_avg = X.mean(axis=1)
        log_data["gradNrm"][k] = float(np.linalg.norm(Y.mean(axis=1)))
        log_data["cons"][k]    = float(np.linalg.norm(X - x_avg[:, None], "fro") / np.sqrt(N))
        log_data["combo"][k]   = log_data["gradNrm"][k] + log_data["cons"][k]
        log_data["ValueF"][k]  = float(np.mean([fi(x_avg) for fi in f]))
        log_data["xBar"][k]    = x_avg
        log_data["commCost"][k] = comm_total / 1024 ** 2
        log_data["timeCost"][k] = time.perf_counter() - t_start
        if norm_x0 is not None:
            log_data["relX"][k] = np.linalg.norm(x_avg - prm["x_opt"]) / norm_x0
        if f_gap is not None:
            log_data["relF"][k] = abs(log_data["ValueF"][k] - prm["f_opt"]) / f_gap
        if k > 0 and not np.isnan(log_data["gradNrm"][k - 1]):
            log_data["rk"][k] = log_data["gradNrm"][k] / max(1e-12, log_data["gradNrm"][k - 1])

        if prm["verbose"] and (k + 1) % print_freq == 0:
            print_progress(k + 1, log_data["gradNrm"][k], log_data["cons"][k],
                           log_data["ValueF"][k],
                           relX=log_data["relX"][k], relF=log_data["relF"][k])

        # stopping
        if prm["tolType"] == "combo" and log_data["combo"][k] < prm["tol"]:
            break
        if prm["tolType"] == "relX" and norm_x0 is not None and log_data["relX"][k] < prm["tol"]:
            break
        if prm["tolType"] == "relF" and f_gap is not None and log_data["relF"][k] < prm["tol"]:
            break

        # ── DIGing update ──────────────────────────────────────────────────────
        alpha_k = prm["alpha"] / np.sqrt(k + 1) if prm["decay_alpha"] else prm["alpha"]
        Wk = get_W(prm["W"], k)
        Mix = np.kron(Wk, np.eye(d))

        # (1) primal update: x^{k+1} = W x^k - α y^k
        Xvec = X.ravel(order="F")
        Yvec = Y.ravel(order="F")
        for _ in range(prm["NC"]):
            Xvec = Mix @ Xvec
        X_new = Xvec.reshape(d, N, order="F") - alpha_k * Y
        if prm["countComm"]:
            comm_total += compute_comm_cost(d, Wk, "vector", rounds=prm["NC"])

        # (2) gradient increment
        G_prev = G.copy()
        for i in range(N):
            G[:, i] = approx_grad(f[i], X_new[:, i])
        dG = G - G_prev

        # (3) gradient tracker: y^{k+1} = W y^k + ∇F(x^{k+1}) - ∇F(x^k)
        Yvec = Y.ravel(order="F")
        for _ in range(prm["NC"]):
            Yvec = Mix @ Yvec
        Y = Yvec.reshape(d, N, order="F") + dG
        if prm["countComm"]:
            comm_total += compute_comm_cost(d, Wk, "vector", rounds=prm["NC"])

        X = X_new

    log_data = trim_log(log_data, k + 1)
    log_data["fail"] = fail_flag
    log_data["failReason"] = fail_reason
    return X.ravel(order="F"), log_data
