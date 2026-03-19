"""Diagnose ESOM: test different penalty values."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solvers.baselines.esom import esom
from problems.obj_factory import obj_factory
from problems.init_policy import init_policy
from utils.helper.graph import generate_random_graph

P = {"Nagent": 10, "p_edge": 0.5, "d_override": 10, "info": 2, "NC": 3}
param_bank, M_alpha_policy, x0_gen = init_policy("regular")

for obj_name in ["ridge", "logsumexp"]:
    args = param_bank.get(obj_name, [P["d_override"]])
    args = [P["d_override"]] + list(args[1:])
    fun_list, d, L_vec, x_opt_list, f_opt_list, is_convex, fname, fparam = \
        obj_factory(obj_name, P["Nagent"], *args)

    np.random.seed(100)
    _, W = generate_random_graph(P["Nagent"], P["p_edge"])
    x0_fn = x0_gen.get(obj_name, lambda d, far: np.random.randn(d))
    x0 = x0_fn(d, False)
    policy = M_alpha_policy[obj_name]
    M_val = policy["M_factor"] * float(L_vec.max())
    alp_val = policy["alpha"] / float(L_vec.max())

    print(f"\n{'='*60}\n  ESOM on {obj_name}  (M_val={M_val:.2f}, L_max={float(L_vec.max()):.1f})")
    print(f"{'='*60}")

    for penalty in [0.5, 1.0, 5.0, M_val]:
        for unit_reg in [0.0, 1.0]:
            prm = dict(Nagent=P["Nagent"], dim=d, f=fun_list, W=W,
                       x_opt=x_opt_list[0], f_opt=float(np.mean(f_opt_list)),
                       fname=fname, fparam=fparam, info=2, NC=P["NC"],
                       alpha=alp_val, M=M_val, decay_alpha=False,
                       esom_penalty=penalty, esom_unit_reg=unit_reg,
                       maxIt=500, tol=1e-12, tolType="combo",
                       verbose=False, countComm=True)
            try:
                _, out = esom(x0.copy(), prm)
                c = np.asarray(out["combo"]); c = c[np.isfinite(c)]
                rf = np.asarray(out["relF"]); rf = rf[np.isfinite(rf)]
                fc = float(c[-1]) if len(c) else float('nan')
                fr = float(rf[-1]) if len(rf) else float('nan')
                steps = len(c)
                tag = 'OK' if fc < 1e-8 else ('DIVG' if fc > 10 else 'partial')
                print(f"  penalty={penalty:5.1f}  unit_reg={unit_reg:.1f}  "
                      f"steps={steps:4d}  combo={fc:.2e}  relF={fr:.2e}  [{tag}]")
            except Exception as e:
                print(f"  penalty={penalty:5.1f}  unit_reg={unit_reg:.1f}  CRASH: {e}")
print("\nDone.")
