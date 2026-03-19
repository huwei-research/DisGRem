"""
Quick smoke test: run all MainComp algorithms on ridge and logsumexp.
Checks that all algorithms converge without crashes.
"""
import numpy as np, sys
sys.path.insert(0, '.')
from utils.alg.alg_bank import get_alg_bank
from utils.helper.graph import generate_random_graph
from problems.obj_factory import obj_factory

N, d = 6, 5
np.random.seed(0)
W, _ = generate_random_graph(N, 0.7)

mc = get_alg_bank('MainComp')
objectives = ['ridge', 'logsumexp']

# Base prm (alpha will be set per-function)
def run_test(fname, alpha_val, M_factor=0.1):
    result = obj_factory(fname, N, d)
    f_list = result[0]
    x_opt = result[3][0]
    f_opt = float(result[4][0])
    fparam = result[7]
    L_max = float(result[2].max())
    alpha = alpha_val / max(L_max, 1e-3)
    np.random.seed(42)
    x0 = np.random.randn(d) * 0.1
    M_val = M_factor * L_max

    print(f"\n{'='*70}")
    print(f"  {fname}  (N={N}, d={d}, alpha={alpha:.4g}, M={M_val:.3g}, L_max={L_max:.3g})")
    print(f"{'='*70}")

    for name, fn in mc:
        prm = dict(Nagent=N, dim=d, f=f_list, W=W, x_opt=x_opt, f_opt=f_opt,
                   alpha=alpha, M=M_val, esom_penalty=M_val,
                   nt_cons_weight=1.0, nt_max_step=5.0,
                   info=2, fname=fname, fparam=fparam,
                   maxIt=300, tol=1e-8, tolType='combo',
                   verbose=False, countComm=True, NC=1)
        try:
            _, log = fn(x0.copy(), prm)
            c = log['combo']; c = c[~np.isnan(c)]
            rf = log['relF']; rf = rf[~np.isnan(rf)]
            steps = len(c)
            fin_combo = float(c[-1]) if steps else float('nan')
            fin_rf = float(rf[-1]) if len(rf) else float('nan')
            status = 'OK' if fin_combo < 1e-6 else ('DIVG' if fin_combo > 10 else 'partial')
            print(f"  {name:20s}  steps={steps:4d}  combo={fin_combo:.2e}  relF={fin_rf:.2e}  [{status}]")
        except Exception as e:
            print(f"  {name:20s}  CRASH: {e}")

run_test('ridge', 0.2, M_factor=0.1)
run_test('logsumexp', 0.4, M_factor=20.0)
print("\nSmoke test done.")
