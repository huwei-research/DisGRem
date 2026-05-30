"""Quick test: can DisGrem converge on quadbad with more iterations?"""
import numpy as np, sys, os
_root = os.getcwd()
sys.path.insert(0, _root)
from utils.alg.alg_bank import get_alg_bank
from problems.obj_factory import obj_factory
from problems.init_policy import init_policy
from utils.helper.graph import generate_random_graph

N, d = 10, 10
np.random.seed(42)
_, W = generate_random_graph(N, 0.5)
fun_list, d_out, L_vec, x_opt_list, f_opt_list, is_convex, fname, fparam = \
    obj_factory('quadbad', N, d)

param_bank, M_alpha_policy, x0_generator = init_policy("regular")
policy = M_alpha_policy.get("quadbad", {"M_factor": 1.0, "alpha": 0.1, "decay": False, "maxIt": 2000})
M_val   = policy["M_factor"] * float(L_vec.max())
alp_val = policy["alpha"] / float(L_vec.max())

np.random.seed(100)
x0_gen = x0_generator.get("quadbad", lambda d, far: np.random.randn(d))
np.random.seed(100)
x0 = x0_gen(d, False)  # shape (d,) - algorithm tiles internally

prm = {
    "W": W, "maxIt": 10000, "tol": 1e-8, "tolType": "relF",
    "NC": 3, "verbose": False, "M": M_val, "alpha": alp_val,
    "decay_alpha": policy["decay"], "f_opt": float(np.mean(f_opt_list)),
    "x_opt": x_opt_list[0] if x_opt_list else None,
    "esom_penalty": 1.0, "Nagent": N, "dim": d_out,
    "f": None,  # set below
}

ab = {name: func for name, func in get_alg_bank("All")}
prm["f"] = fun_list
for alg_name in ["DisGrem", "CeDisGrem", "DQM"]:
    _, out = ab[alg_name](x0.copy(), dict(prm))
    combo_vals = np.array(out.get('combo', [1]))
    relF_vals  = np.array(out.get('relF', [1]))
    relX_vals  = np.array(out.get('relX', [1]))
    n_steps = len(combo_vals)
    print(f"\n[{alg_name}] ran {n_steps} logged steps:")
    print(f"  Final combo={combo_vals[-1]:.4e}  relF={relF_vals[-1]:.4e}  relX={relX_vals[-1]:.4e}")
    print(f"  Min combo={np.nanmin(combo_vals):.4e}")
    for frac in [0.2, 0.5, 0.8, 1.0]:
        idx = int(n_steps*frac) - 1
        if 0 <= idx < n_steps:
            print(f"  At {int(frac*100)}% ({idx+1} steps): combo={combo_vals[idx]:.4e}")
