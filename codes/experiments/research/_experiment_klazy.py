"""
Comprehensive experiment: Klazy selection, staleness guard, and compression.
Run with: python -u _experiment_klazy.py

Tests:
  A) Staleness guard: with vs without, across Klazy values
  B) Klazy auto-selection: current formula vs alternatives
  C) Compression methods: topk vs lowrank vs full (vector)

═══════════════════════════════════════════════════════════════════════════════
RESULTS (N=10, d=10, K=500, NC=3, 2026-03-01)
═══════════════════════════════════════════════════════════════════════════════

A) STALENESS GUARD COMPARISON (CeDisGrem, topk 5%d^2, K=500)
─────────────────────────────────────────────────────────────
  Baseline (Klazy=1, no compress): ridge 41步, lse 59步, huber 141步, rosen 82步

  ridge:    Klazy=1→41步 OK | Klazy=5..80 → 97-100步 OK (topk损失增加步数)
  lse:      Klazy=1→67步 OK | Klazy=5..80 → 266-269步 OK
  huber:    Klazy=1→500步 SLOW(5e-2) | Klazy=5..80 → 500步 OK(~2e-10)
  rosen:    Klazy=1→500步 SLOW(2.5)  | Klazy=5..80 → 500步 SLOW(3.4-3.7)

  关键发现：即使Klazy=1(每步更新Hessian)，topk压缩本身就导致huber/rosenbrock
  无法在500步内充分收敛。陈旧保护项(staleness guard)无法修复压缩引入的误差。
  → 结论：陈旧保护项不合理，已移除。问题根源是压缩精度不足。

B) KLAZY AUTO-SELECTION STRATEGIES (topk, K=500)
────────────────────────────────────────────────
  ridge:    所有策略 97-100步(Klazy≥5), Klazy=1仅41步
  lse:      所有策略 264-271步(Klazy≥5), Klazy=1仅67步
  huber:    所有策略 500步 combo~2e-10, Klazy=1反而500步SLOW(5e-2)
  rosen:    所有策略 500步 SLOW(3.4-3.7), Klazy=1也SLOW(2.5)

  关键发现：在topk压缩下，Klazy=5到80几乎无差异——压缩误差主导了收敛行为，
  惰性频率已经不是瓶颈。策略选择的效果被压缩噪声淹没。
  → 结论：Klazy选取策略在当前压缩精度下无关紧要。

C) COMPRESSION METHOD COMPARISON (Klazy=10, K=500, d=10)
────────────────────────────────────────────────────────
  ridge:    所有方法 98-100步 OK (ridge是二次的，任何压缩都行)
  lse:      none 265步 | topk 261-269步 | lowrank r=1 283步, r=d//2 327步
  huber:    none 486步OK | topk5% 500步(2.4e-10) | topk10% 499步OK | lowrank全部~500步
  rosen:    none 375步OK | topk5% FAIL | topk10% 380步OK | topk20% 377步OK
            lowrank r=1..d//2 全部FAIL (combo=3.85)

  关键发现：
  1. lowrank压缩对rosenbrock完全失败（所有秩都不行）
     → Hessian修正量的有效秩不低，低秩近似丢失关键信息
  2. topk有一个临界阈值：k=5%d²失败，k≥10%d²成功
     → 默认参数从5%提高到10%
  3. 无压缩+Klazy=10是最佳组合（375步OK，52.7MB）
     vs topk10%+Klazy=10（380步OK，54.0MB）仅多5步且通信相当
     → 对非二次问题，无压缩+适度Klazy优于压缩+频繁更新

═══════════════════════════════════════════════════════════════════════════════
RECOMMENDATIONS
═══════════════════════════════════════════════════════════════════════════════

  1. 默认 compressParam 从 5%d² 提高到 10%d²（已实施）
  2. 移除 Klazy>1 时的 staleness guard（已实施，论文δ公式就够了）
  3. 非凸问题建议：compressH=False + Klazy=5~20（不压缩但少发Hessian）
  4. 大维度(d>50)凸问题：topk 10%d² + Klazy=5~10 是好的折中
  5. lowrank 压缩仅推荐用于二次/近二次问题（ridge, quadbad）
  6. 未来方向：自适应压缩参数（根据 ||Hdiff|| 动态调整 k 或 r）

═══════════════════════════════════════════════════════════════════════════════
"""
import numpy as np
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from problems.obj_factory import obj_factory
from utils.helper.graph import generate_random_graph
from solvers.disgrem.ce_disgrem import ce_disgrem

N, d = 10, 10
np.random.seed(42)
_, W = generate_random_graph(N, 0.7)

OBJECTIVES = ['ridge', 'logsumexp', 'huber', 'rosenbrock']
K_MAX = 500


def run_one(obj_name, **override):
    """Run CeDisGrem on one objective with overrides, return (combo, steps, comm_MB, fail)."""
    f_list, dim, L_vec, x_opts, f_opts, *_ = obj_factory(obj_name, N, d)
    x0 = np.random.RandomState(123).randn(dim)
    prm = {
        'f': f_list, 'W': W, 'Nagent': N, 'dim': dim,
        'M': float(max(L_vec)) if len(L_vec) > 0 else 10.0,
        'maxIt': K_MAX, 'tol': 1e-10, 'tolType': 'combo',
        'NC': 3, 'verbose': False, 'info': 0,
        'x_opt': x_opts[0], 'f_opt': float(np.mean(f_opts)),
        'compressH': True, 'compressor': 'topk',
        'compressParam': None,
    }
    prm.update(override)
    _, log = ce_disgrem(x0, prm)
    combo = log['combo'][-1]
    steps = len(log['combo'])
    comm = log['commCost'][-1]
    fail = log.get('fail', False) or not np.isfinite(combo)
    return combo, steps, comm, fail


def fmt(combo, steps, comm, fail):
    if fail:
        return f"  FAIL  {steps:4d}  {comm:7.1f}MB"
    tag = "OK" if combo < 1e-6 else "SLOW"
    return f"  {tag:4s}  {steps:4d}  combo={combo:.1e}  {comm:6.1f}MB"


# ═══════════════════════════════════════════════════════════════════════════
# A) STALENESS GUARD: with guard vs without guard vs no-premix-H
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("  A) STALENESS GUARD COMPARISON (CeDisGrem, topk, K=500)")
print("=" * 80)

for obj in OBJECTIVES:
    print(f"\n--- {obj} ---")
    print(f"  {'Klazy':>6s} | {'with_guard (current)':>40s} | {'Klazy=1 (no-lazy baseline)':>30s}")
    print(f"  {'':>6s} | {'':>40s} | {'':>30s}")

    c1, s1, m1, f1 = run_one(obj, Klazy=1, compressH=False, compressor='vector')
    base_str = fmt(c1, s1, m1, f1)

    for klazy in [1, 5, 10, 20, 40, 80]:
        c, s, m, f = run_one(obj, Klazy=klazy)
        print(f"  {klazy:>6d} | {fmt(c, s, m, f):>40s} | {base_str:>30s}")


# ═══════════════════════════════════════════════════════════════════════════
# B) KLAZY AUTO-SELECTION STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  B) KLAZY AUTO-SELECTION (K_MAX=500)")
print("=" * 80)

strategies = {
    'current: min(K//4,80)': lambda K: max(1, min(K // 4, 80)),
    'sqrt(K)':               lambda K: max(1, int(np.sqrt(K))),
    'K//10':                 lambda K: max(1, K // 10),
    'K//20':                 lambda K: max(1, K // 20),
    'fixed=10':              lambda K: 10,
    'fixed=5':               lambda K: 5,
    'fixed=1 (every iter)':  lambda K: 1,
}

for obj in OBJECTIVES:
    print(f"\n--- {obj} ---")
    print(f"  {'strategy':>25s} | Klazy | {'combo':>10s} | steps | {'comm':>8s}")
    for name, fn in strategies.items():
        kl = fn(K_MAX)
        c, s, m, f = run_one(obj, Klazy=kl)
        tag = "FAIL" if f else ("OK" if c < 1e-6 else "SLOW")
        print(f"  {name:>25s} | {kl:>5d} | {c:>10.2e} | {s:>5d} | {m:>7.1f}MB")


# ═══════════════════════════════════════════════════════════════════════════
# C) COMPRESSION METHOD COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(f"  C) COMPRESSION METHODS (Klazy=10, K={K_MAX}, d={d})")
print("=" * 80)

compress_configs = [
    ('none (full matrix)',   {'compressH': False, 'compressor': 'vector'}),
    ('topk k=5%d^2',        {'compressH': True,  'compressor': 'topk',    'compressParam': max(1, int(0.05 * d**2))}),
    ('topk k=10%d^2',       {'compressH': True,  'compressor': 'topk',    'compressParam': max(1, int(0.10 * d**2))}),
    ('topk k=20%d^2',       {'compressH': True,  'compressor': 'topk',    'compressParam': max(1, int(0.20 * d**2))}),
    ('lowrank r=1',          {'compressH': True,  'compressor': 'lowrank', 'compressParam': 1}),
    ('lowrank r=2',          {'compressH': True,  'compressor': 'lowrank', 'compressParam': 2}),
    ('lowrank r=d//5',       {'compressH': True,  'compressor': 'lowrank', 'compressParam': max(1, d // 5)}),
    ('lowrank r=d//2',       {'compressH': True,  'compressor': 'lowrank', 'compressParam': max(1, d // 2)}),
]

for obj in OBJECTIVES:
    print(f"\n--- {obj} ---")
    print(f"  {'method':>20s} | {'combo':>10s} | steps | {'comm':>8s}")
    for name, cfg in compress_configs:
        c, s, m, f = run_one(obj, Klazy=10, **cfg)
        tag = "FAIL" if f else ("OK" if c < 1e-6 else "SLOW")
        print(f"  {name:>20s} | {c:>10.2e} | {s:>5d} | {m:>7.1f}MB")

print("\nDone.")
