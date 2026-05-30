"""
extra.py - EXTRA: An Exact First-Order Algorithm for Decentralized Consensus Optimization.

Reference:
    Shi, W., Ling, Q., Wu, G., & Yin, W. (2015).
    EXTRA: An exact first-order algorithm for decentralized consensus optimization.
    SIAM Journal on Optimization, 25(2), 944-966.

Iteration (vectorised over N agents, X is dxN):
    W_half = (I + W) / 2
    x^1    = W_half x^0 - alpha gradF(x^0)
    x^{k+1}= x^k + W x^k - W_half x^{k-1} - alpha (gradF(x^k) - gradF(x^{k-1}))

Communication per iteration: one vector-exchange round (cheaper than DGD with tracking).
Converges linearly to exact consensus for strongly convex, smooth objectives.
"""

from __future__ import annotations
import time
import numpy as np
from utils.helper.log_utils import init_log, trim_log, compute_comm_cost, print_progress
from utils.helper.eval_utils import approx_grad
from utils.helper.graph import get_W


def extra(x0: np.ndarray, prm: dict):
    """
    EXTRA algorithm.

    prm extra fields:
        .alpha        stepsize (required)
        .decay_alpha  bool - use alpha/√k  [False]
        .NC           consensus rounds applied to W  [1]
    """
    p = {
        "maxIt": 200, "tol": 1e-8, "tolType": "combo",
        "verbose": True, "alpha": None, "decay_alpha": False,
        "countComm": True, "memoryLimitMB": np.inf, "NC": 1,
        "x_opt": None, "f_opt": None,
        # EXTRA-specific override: if 'alpha_extra' is set it takes priority over 'alpha'
        "alpha_extra": None,
    }
    p.update(prm)
    prm = p
    assert prm["alpha"] is not None, "stepsize alpha is required for EXTRA"
    # Step-size policy for EXTRA:
    #   If an algorithm-specific override 'alpha_extra' is supplied, use it directly.
    #   Otherwise use the base alpha unchanged - EXTRA can use the same step size
    #   as other first-order methods (alpha <= 1/L) and still converge linearly.
    #   The previous 5x auto-scaling caused divergence when alpha was already near 1/L.
    if prm["alpha_extra"] is not None:
        prm["alpha"] = float(prm["alpha_extra"])

    N = prm["Nagent"]; d = prm["dim"]; f = prm["f"]
    K = prm["maxIt"]
    print_freq = max(1, K // 25)

    # ── weight matrices ────────────────────────────────────────────────────────
    Wk = get_W(prm["W"], 0)
    # EXTRA uses W_half = (I + W)/2; apply NC rounds by computing W^NC
    Wk_nc = np.linalg.matrix_power(Wk, prm["NC"]) if prm["NC"] > 1 else Wk
    W_half = (np.eye(N) + Wk_nc) / 2          # (N, N)
    Mix     = np.kron(Wk_nc, np.eye(d))        # (dN, dN)  ≡  X → X @ W_nc
    Mix_half = np.kron(W_half, np.eye(d))      # (dN, dN)  ≡  X → X @ W_half

    # ── initialise ─────────────────────────────────────────────────────────────
    X = np.tile(x0.ravel()[:, None], (1, N))   # (d, N)
    G = np.zeros((d, N))
    for i in range(N):
        G[:, i] = approx_grad(f[i], x0.ravel())

    log_data = init_log(K)
    log_data["xBar"] = np.full((K, d), np.nan)

    x_avg0 = X.mean(axis=1)
    norm_x0, f_gap = None, None
    if prm["x_opt"] is not None and np.all(np.isfinite(prm["x_opt"])):
        norm_x0 = max(np.linalg.norm(x_avg0 - prm["x_opt"]), 1e-12)
    if prm["f_opt"] is not None and np.isfinite(prm["f_opt"]):
        f0 = float(np.mean([fi(x_avg0) for fi in f]))
        f_gap = max(abs(f0 - prm["f_opt"]), 1e-12)

    # ── first EXTRA step  (k=0 → k=1) ─────────────────────────────────────────
    X_prev = X.copy()
    G_prev = G.copy()
    Xvec = Mix_half @ X.ravel(order="F") - prm["alpha"] * G.ravel(order="F")
    X = Xvec.reshape(d, N, order="F")
    G_curr = np.zeros((d, N))
    for i in range(N):
        G_curr[:, i] = approx_grad(f[i], X[:, i])

    comm_total = 0.0
    if prm["countComm"]:
        comm_total += compute_comm_cost(d, Wk_nc, "vector", rounds=1)

    fail_flag = False; fail_reason = ""
    t_start = time.perf_counter()

    # ── main loop (k=1, 2, …) ─────────────────────────────────────────────────
    k = 0
    for k in range(K):
        if not np.all(np.isfinite(X)) or np.any(np.abs(X) > 1e12):
            fail_flag = True; fail_reason = "numerical overflow"; break

        # Logging uses current X (before update) -  consistent with other algs
        x_avg = X.mean(axis=1)
        grad_avg = G_curr.mean(axis=1)
        log_data["gradNrm"][k] = float(np.linalg.norm(grad_avg))
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

        # ── EXTRA update ──────────────────────────────────────────────────────
        # x^{k+1} = x^k + W x^k - W_half x^{k-1} - alpha(g^k - g^{k-1})
        alpha_k = prm["alpha"] / np.sqrt(k + 1) if prm["decay_alpha"] else prm["alpha"]
        X_prev_vec = X_prev.ravel(order="F")
        X_curr_vec = X.ravel(order="F")
        dG = (G_curr - G_prev).ravel(order="F")
        X_next_vec = (X_curr_vec
                      + Mix @ X_curr_vec
                      - Mix_half @ X_prev_vec
                      - alpha_k * dG)

        # update state
        X_prev = X.copy()
        G_prev = G_curr.copy()
        X = X_next_vec.reshape(d, N, order="F")
        for i in range(N):
            G_curr[:, i] = approx_grad(f[i], X[:, i])

        if prm["countComm"]:
            comm_total += compute_comm_cost(d, Wk_nc, "vector", rounds=1)

    log_data = trim_log(log_data, k + 1)
    log_data["fail"] = fail_flag
    log_data["failReason"] = fail_reason
    return X.ravel(order="F"), log_data
