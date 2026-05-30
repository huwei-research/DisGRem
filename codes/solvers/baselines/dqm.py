"""
dqm.py - DQM: Decentralised Quadratically Approximated ADMM.
Reference: Mokhtari, Shi, Ling, Ribeiro, IEEE TSP 2016 (arXiv:1510.07356).

Algorithm 1 (paper, at node i):
  x_{i,k+1} = (2c*d_i*I + H_i(x_{i,k}))^{-1}
               * (c*d_i*x_{i,k} + c*sum_{j in N_i} x_{j,k}
                  + H_i(x_{i,k})*x_{i,k} - g_i(x_{i,k}) - phi_{i,k})
  phi_{i,k+1} = phi_{i,k} + c * sum_{j in N_i} (x_{i,k+1} - x_{j,k+1})

where:
  d_i  = |N_i| = strict graph degree (neighbours only, NOT counting self)
  N_i  = set of strict neighbours (j != i, w_ij > 0)
  c    = ADMM penalty coefficient

Communication: one round per iteration (exchange x_new with neighbours).
Previous implementation added un-documented pre/post-mixing rounds and used
a degree that incorrectly counted self-loops (w_ii > 0).  Both issues are
corrected here.
"""

from __future__ import annotations
import time
import numpy as np
from utils.helper.log_utils import init_log, trim_log, compute_comm_cost, print_progress
from utils.helper.eval_utils import compute_info
from utils.helper.graph import get_W


def dqm(x0: np.ndarray, prm: dict):
    """
    DQM algorithm (paper-accurate implementation).

    Extra prm fields:
        .c     ADMM penalty coefficient  [0.5]
        .epsl  ridge for SPD             [1e-3]
    """
    p = {
        "c": 0.5, "epsl": 1e-3,
        "maxIt": 200, "tol": 1e-8, "tolType": "combo",
        "verbose": True, "countComm": True, "memoryLimitMB": np.inf,
        "info": 0, "NC": 1,
        "x_opt": None, "f_opt": None,
    }
    p.update(prm)
    prm = p

    N = prm["Nagent"]; d = prm["dim"]; f = prm["f"]
    K = prm["maxIt"]
    c = prm["c"]
    print_freq = max(1, K // 25)

    X = np.tile(x0.ravel()[:, None], (1, N))
    Phi = np.zeros((d, N))   # dual variables

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

        # Strict neighbour masks and degrees (paper uses d_i = |N_i|, j != i)
        nbr_masks = [(Wk[i] > 0) & (np.arange(N) != i) for i in range(N)]
        deg = np.array([m.sum() for m in nbr_masks], dtype=float)

        # (1) local grad & Hessian at current x (no pre-mixing)
        G = np.zeros((d, N)); H = np.zeros((d, d, N))
        for i in range(N):
            gi, Hi = compute_info(X[:, i], prm, i, "both")
            G[:, i] = gi; H[:, :, i] = Hi

        # (2) primal update (paper eq. 14)
        X_new = np.zeros((d, N))
        for i in range(N):
            nbr = nbr_masks[i]
            sum_nbr = X[:, nbr].sum(axis=1)          # sum_{j in N_i} x_j
            rhs = (c * deg[i] * X[:, i] + c * sum_nbr
                   + H[:, :, i] @ X[:, i] - G[:, i] - Phi[:, i])
            M_mat = (2 * c * deg[i] * np.eye(d) + H[:, :, i]
                     + prm["epsl"] * np.eye(d))
            try:
                X_new[:, i] = np.linalg.solve(M_mat, rhs)
            except np.linalg.LinAlgError:
                X_new[:, i] = np.linalg.lstsq(M_mat, rhs, rcond=None)[0]

        # (3) exchange x_new with neighbours - one communication round
        if prm["countComm"]:
            comm_total += compute_comm_cost(d, Wk, "vector", rounds=prm["NC"])

        # (4) dual update (paper eq. 15)
        for i in range(N):
            nbr = nbr_masks[i]
            # phi_{i,k+1} = phi_{i,k} + c * sum_{j in N_i} (x_{i,k+1} - x_{j,k+1})
            Phi[:, i] += c * (deg[i] * X_new[:, i] - X_new[:, nbr].sum(axis=1))

        X = X_new
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
