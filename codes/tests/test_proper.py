"""
Proper test: run MainComp on ridge only using EXACT same parameter setup as run_regular.py.
"""
import os
import sys
import time
from pathlib import Path

_CODES_ROOT = Path(__file__).resolve().parents[1]
if str(_CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODES_ROOT))

import numpy as np

from utils.alg.alg_bank import get_alg_bank
from problems.obj_factory import obj_factory
from problems.init_policy import init_policy
from utils.helper.graph import generate_random_graph

P = {
    "Nagent":      10, "p_edge": 0.5, "maxIt": 500, "tol": 1e-12,
    "tolType":     "combo", "verbose": False, "far": False, "useWorst": False,
    "nStart":      1, "d_override": 10, "info": 2, "NC": 3,
    "countComm":   True, "nt_max_step": 5.0,
}

alg_bank = get_alg_bank("MainComp")
param_bank, M_alpha_policy, x0_generator = init_policy("regular")

for obj_name in ["ridge", "logsumexp", "linlog"]:
    args = param_bank.get(obj_name, [P["d_override"]])
    if args and isinstance(args[0], (int, float)):
        args = [P["d_override"]] + list(args[1:])

    fun_list, d, L_vec, x_opt_list, f_opt_list, is_convex, fname, fparam = \
        obj_factory(obj_name, P["Nagent"], *args)

    np.random.seed(100)
    _, W = generate_random_graph(P["Nagent"], P["p_edge"])
    x0_gen = x0_generator.get(obj_name, lambda d, far: np.random.randn(d))
    x0 = x0_gen(d, P["far"])

    policy = M_alpha_policy.get(obj_name, {"M_factor": 1.0, "alpha": 0.1,
                                            "decay": False, "maxIt": P["maxIt"]})
    maxIt_obj = policy.get("maxIt", P["maxIt"])
    M_val   = policy["M_factor"] * float(L_vec.max())
    alp_val = policy["alpha"]    / float(L_vec.max())

    prm = dict(P)
    prm.update({
        "f": fun_list, "fname": fname, "fparam": fparam, "dim": d,
        "M": M_val, "alpha": alp_val, "decay_alpha": policy["decay"],
        "maxIt": maxIt_obj, "tolType": policy.get("tolType", "combo"),
        "x_opt": x_opt_list[0] if x_opt_list else None,
        "f_opt": float(np.mean(f_opt_list)),
        "W": W, "esom_penalty": 1.0, "nt_cons_weight": 1.0,
    })

    f0_val = float(np.mean([fi(x0) for fi in fun_list]))
    print(f"\n{'='*65}")
    print(f"  {obj_name}  (N={P['Nagent']}, d={d}, alpha={alp_val:.4g},"
          f" M={M_val:.3g}, maxIt={maxIt_obj})")
    print(f"{'='*65}")

    for alg_name, alg_func in alg_bank:
        t0 = time.perf_counter()
        try:
            _, out = alg_func(x0.copy(), dict(prm))
            elapsed = time.perf_counter() - t0
            c  = np.asarray(out.get("combo", []), dtype=float)
            rf = np.asarray(out.get("relF",  []), dtype=float)
            c  = c[np.isfinite(c)]; rf = rf[np.isfinite(rf)]
            fc = float(c[-1]) if len(c) else float('nan')
            fr = float(rf[-1]) if len(rf) else float('nan')
            steps = len(c)
            if not np.isfinite(fc) or fc > 10:
                status = 'DIVG'
            elif fc < 1e-8:
                status = 'OK'
            else:
                status = 'partial'
            print(f"  {alg_name:20s}  steps={steps:4d}  combo={fc:.2e}  "
                  f"relF={fr:.2e}  [{status}]  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  {alg_name:20s}  CRASH: {e}")

print("\nTest done.")
