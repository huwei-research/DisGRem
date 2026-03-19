"""
network_giant.py – Network-GIANT: Fully Distributed Newton-Type Optimisation
via Harmonic Hessian Consensus.

Reference:
  Xie, Johansson, "Network-GIANT: Fully Distributed Newton-Type Optimization
  via Harmonic Hessian Consensus", IEEE ICASSP 2024 (arXiv:2305.07898).

Algorithm (per iteration):
  1. Mixing:     x̃_i  = Σ_j w_ij x_j^k          (primal consensus)
  2. GT update:  y_i^{k+1} = Σ_j w_ij [y_j^k + ∇f_j(x̃_j) - ∇f_j(x_i^k)]
  3. Newton dir: n_i  = (H_i(x̃_i) + εI)^{-1} y_i^{k+1}
  4. Dir avg:    n̄_i  = Σ_j w_ij n_j             (harmonic Hessian consensus)
  5. Update:     x_i^{k+1} = x̃_i - α * n̄_i

Communication: 3 rounds per iteration (mix x, mix y, mix n).

The "harmonic Hessian" consensus averages H_i^{-1}g_i across neighbours,
approximating (mean_H)^{-1}(mean_g) via the harmonic mean inequality.

Parameters:
    alpha_step  step size for the Newton update          [1.0]
    epsl        Hessian regularisation                   [1e-4]
"""

from __future__ import annotations
import time
import numpy as np
from utils.helper.log_utils import init_log, trim_log, compute_comm_cost, print_progress
from utils.helper.eval_utils import compute_info
from utils.helper.graph import get_W


def network_giant(x0: np.ndarray, prm: dict):
    """Network-GIANT harmonic Hessian consensus Newton algorithm."""
    p = {
        "alpha_step": 1.0,  # Newton step size (1.0 = full Newton near optimum)
        "epsl": 1e-4,       # Hessian regularisation (prevents singular H)
        "maxIt": 200, "tol": 1e-8, "tolType": "combo",
        "verbose": True, "countComm": True, "memoryLimitMB": np.inf,
        "info": 0, "NC": 1,
        "x_opt": None, "f_opt": None,
    }
    p.update(prm)
    prm = p

    N = prm["Nagent"]; d = prm["dim"]; f = prm["f"]
    K = prm["maxIt"]
    alpha_s = prm["alpha_step"]
    epsl = prm["epsl"]
    # Hessian regularisation: use "ng_epsl" if provided, else fall back to epsl.
    # Do NOT use M (DisGrem's proximal parameter) — that is typically too large and
    # would shrink Newton steps, causing NetworkGIANT to converge to a biased point.
    epsl_H = prm.get("ng_epsl", epsl)
    print_freq = max(1, K // 25)

    X = np.tile(x0.ravel()[:, None], (1, N))

    # Initialise gradient tracker Y = local gradients at x0
    Y = np.zeros((d, N))
    for i in range(N):
        gi, _ = compute_info(X[:, i], prm, i, "grad")
        Y[:, i] = gi

    # G_prev is not needed with the corrected GT; kept for compatibility (unused).
    # G_prev = Y.copy()  # removed — see GT update below

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
        NC = prm["NC"]

        # (1) Primal consensus: x̃_i = Σ_j w_ij x_j
        Xvec = X.ravel(order="F")
        for _ in range(NC):
            Xvec = Mix @ Xvec
        X_tilde = Xvec.reshape(d, N, order="F")
        if prm["countComm"]:
            comm_total += compute_comm_cost(d, Wk, "vector", rounds=NC)

        # Snapshot gradient at x^k (before the update) for the GT correction.
        G_curr = np.zeros((d, N))
        for i in range(N):
            gi, _ = compute_info(X[:, i], prm, i, "grad")
            G_curr[:, i] = gi

        # (2) Gradient tracking update (DIGing-style, captures full update)
        # y_i^{k+1} = Σ_j w_ij [y_j^k + ∇f_j(x_j^{k+1}) - ∇f_j(x_j^k)]
        # This is applied AFTER the Newton step (see below).
        # G_tilde needed for Newton direction computation only (not GT).
        G_tilde = np.zeros((d, N))
        for i in range(N):
            gi, _ = compute_info(X_tilde[:, i], prm, i, "grad")
            G_tilde[:, i] = gi

        # (3) Local Newton directions: n_i = (H_i(x̃_i) + ε_H·I)^{-1} y_i
        # Use CURRENT y (before GT update) since x̃ is computed at current x.
        # Adaptive LM regularisation: bound step size far from optimum.
        N_dirs = np.zeros((d, N))
        for i in range(N):
            _, Hi = compute_info(X_tilde[:, i], prm, i, "hess")
            tau_lm = max(epsl_H + epsl, float(np.linalg.norm(Y[:, i])))
            D = Hi + tau_lm * np.eye(d)
            try:
                N_dirs[:, i] = np.linalg.solve(D, Y[:, i])
            except np.linalg.LinAlgError:
                N_dirs[:, i] = np.linalg.lstsq(D, Y[:, i], rcond=None)[0]

        # (4) Harmonic Hessian consensus: average Newton directions
        Nvec = N_dirs.ravel(order="F")
        for _ in range(NC):
            Nvec = Mix @ Nvec
        N_bar = Nvec.reshape(d, N, order="F")
        if prm["countComm"]:
            comm_total += compute_comm_cost(d, Wk, "vector", rounds=NC)

        # (5) Primal update: x_i^{k+1} = x̃_i - α * n̄_i
        X_new = X_tilde - alpha_s * N_bar

        # (6) GT update (DIGing-style): capture full gradient change x^k → x^{k+1}
        # y_i^{k+1} = Σ_j w_ij [y_j^k + ∇f_j(x^{k+1}) - ∇f_j(x^k)]
        G_new = np.zeros((d, N))
        for i in range(N):
            gi, _ = compute_info(X_new[:, i], prm, i, "grad")
            G_new[:, i] = gi

        delta_G = G_new - G_curr              # ∇f(x^{k+1}) - ∇f(x^k) per node
        Y_update = Y + delta_G
        Yvec = Y_update.ravel(order="F")
        for _ in range(NC):
            Yvec = Mix @ Yvec
        Y_new = Yvec.reshape(d, N, order="F")
        if prm["countComm"]:
            comm_total += compute_comm_cost(d, Wk, "vector", rounds=NC)

        X = X_new; Y = Y_new

        x_bar = X.mean(axis=1)
        g_avg = Y.mean(axis=1)
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
