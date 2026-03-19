"""
sonata.py – SONATA (Second-Order Surrogate iNcremenTAl distributed optimisation).

Implements the second-order surrogate variant of SONATA from:
  Sun, Scutari, Shi, "Improving the Convergence of Decentralized Gradient Descent
  via Surrogate Objectives", IEEE TSP 2022.

The algorithm is equivalent to "Gradient Tracking with a local Newton step"
(GT-Newton) and is sometimes also called NEXT-II / Newton-GT in the literature.

Update rules (per iteration):
  1. Mixing:     x̃_i  = Σ_j w_ij x_j^k          (consensus on primal)
  2. NT step:    x_i^{k+1} = x̃_i - (H_i(x̃_i) + τI)^{-1} y_i^k
  3. GT update:  y_i^{k+1} = Σ_j w_ij [y_j^k + ∇f_j(x_j^{k+1}) - ∇f_j(x̃_j)]

Communication: 2 rounds per iteration (mix x, mix y).

Parameters:
    tau      regularisation added to local Hessian (ensures step is bounded)  [1e-1]
    epsl     additional small ridge for SPD          [1e-6]
"""

from __future__ import annotations
import time
import numpy as np
from utils.helper.log_utils import init_log, trim_log, compute_comm_cost, print_progress
from utils.helper.eval_utils import compute_info
from utils.helper.graph import get_W


def sonata(x0: np.ndarray, prm: dict):
    """SONATA second-order surrogate algorithm."""
    p = {
        # sonata_tau: Tikhonov regularisation added to the surrogate Hessian.
        # Must be SMALL relative to H (otherwise x barely moves and the tracker
        # converges to 0 at a biased non-optimal point).  Use "sonata_tau" in prm
        # to override; defaults to epsl (≈ pure Newton with minimal regularisation).
        "sonata_tau": None,   # None → use epsl
        "epsl": 1e-4,         # also serves as default tau when sonata_tau is None
        "maxIt": 200, "tol": 1e-8, "tolType": "combo",
        "verbose": True, "countComm": True, "memoryLimitMB": np.inf,
        "info": 0, "NC": 1,
        "x_opt": None, "f_opt": None,
    }
    p.update(prm)
    prm = p

    N = prm["Nagent"]; d = prm["dim"]; f = prm["f"]
    K = prm["maxIt"]
    epsl = prm["epsl"]
    # tau: small Hessian regularisation (like Levenberg-Marquardt).
    # NOT the DisGrem M parameter — SONATA needs tau << H for accurate Newton steps.
    tau = prm.get("sonata_tau", None)
    if tau is None or tau <= 0:
        tau = epsl
    print_freq = max(1, K // 25)

    X = np.tile(x0.ravel()[:, None], (1, N))

    # Initialise gradient tracker Y = local gradients at x0
    Y = np.zeros((d, N))
    for i in range(N):
        gi, _ = compute_info(X[:, i], prm, i, "grad")
        Y[:, i] = gi

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

        # Snapshot gradient at x^k (before any update) for the GT correction.
        G_curr = np.zeros((d, N))
        for i in range(N):
            gi, _ = compute_info(X[:, i], prm, i, "grad")
            G_curr[:, i] = gi

        # (1) Primal consensus mixing: x̃_i = Σ_j w_ij x_j
        Xvec = X.ravel(order="F")
        for _ in range(NC):
            Xvec = Mix @ Xvec
        X_tilde = Xvec.reshape(d, N, order="F")
        if prm["countComm"]:
            comm_total += compute_comm_cost(d, Wk, "vector", rounds=NC)

        # (2) Gradient-tracker consensus mixing: ỹ_i = Σ_j w_ij y_j
        Yvec = Y.ravel(order="F")
        for _ in range(NC):
            Yvec = Mix @ Yvec
        Y_tilde = Yvec.reshape(d, N, order="F")
        if prm["countComm"]:
            comm_total += compute_comm_cost(d, Wk, "vector", rounds=NC)

        # (3) Newton step on surrogate: x_new_i = x̃_i - (H_i(x̃_i) + τI)^{-1} ỹ_i
        # Adaptive Levenberg-Marquardt regularisation: when the gradient tracker y is
        # large (far from optimum), increase tau to bound step size ≤ O(1).  Near the
        # optimum ||y|| → 0, so tau_lm → tau (pure Newton regime).
        X_new = np.zeros((d, N))
        for i in range(N):
            _, Hi = compute_info(X_tilde[:, i], prm, i, "hess")
            tau_lm = max(tau + epsl, float(np.linalg.norm(Y_tilde[:, i])))
            D = Hi + tau_lm * np.eye(d)
            try:
                X_new[:, i] = X_tilde[:, i] - np.linalg.solve(D, Y_tilde[:, i])
            except np.linalg.LinAlgError:
                X_new[:, i] = X_tilde[:, i] - np.linalg.lstsq(D, Y_tilde[:, i], rcond=None)[0]

        # (4) Gradient tracking update (correct SONATA/NEXT formula from Scutari & Sun 2022):
        # y_i^{k+1} = Σ_j w_ij [y_j^k + ∇f_j(x^{k+1}) - ∇f_j(x^k)]
        # Uses G_curr = ∇f(x^k) (pre-mixing) and ∇f(X_new) = ∇f(x^{k+1}).
        G_new = np.zeros((d, N))
        for i in range(N):
            gi_new, _ = compute_info(X_new[:, i], prm, i, "grad")
            G_new[:, i] = gi_new

        delta_G = G_new - G_curr            # ∇f(x^{k+1}) - ∇f(x^k)
        Y_update = Y + delta_G
        Yvec = Y_update.ravel(order="F")
        for _ in range(NC):
            Yvec = Mix @ Yvec
        Y_new = Yvec.reshape(d, N, order="F")
        if prm["countComm"]:
            comm_total += compute_comm_cost(d, Wk, "vector", rounds=NC)

        X = X_new; Y = Y_new

        x_bar = X.mean(axis=1)
        g_avg = Y.mean(axis=1)    # Y tracks the avg gradient
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
