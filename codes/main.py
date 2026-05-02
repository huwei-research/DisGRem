"""
main.py - Unified entry point for DisGrem experiment suite.

USAGE
-----
    python main.py [mode] [arg]

    mode : regular | robust | comm | ada | scale | all (default)
    arg  : mode-specific sub-argument (see below)

MODES
-----
    regular  Main 9-function benchmark (d=30, 20 MC, MainComp algorithms).
             arg = all | ridge | quadbad | convexset | nonconvexset  (default: all)
    robust   Robustness: Part1 starting-point (100MC) + Part2 param sweep.
             arg = all | start | param  (default: all)
    comm     Comm-cost study: Part 1 Ce benefit + Part 2 Klazy/compression ablation.
             arg = all | ce | ablation  (default: all)
    ada      Adaptive mechanism: M trajectory + Ada-vs-fixed-M + init-M.
             (no sub-arguments)
    scale    Dimension scalability study (3 functions x 3 dims x 4 algs x 5 MC).
             (no sub-arguments)
    all      Run all four experiments sequentially (default).

EXAMPLES
--------
    python main.py                  # run all 4 experiments
    python main.py regular          # regular benchmark only (all 9 functions)
    python main.py regular ridge    # regular benchmark, ridge only
    python main.py robust start     # robustness Part 1 only
    python main.py comm             # comm study (Ce benefit + ablation)
    python main.py comm ce          # Part 1 only (Ce benefit)
    python main.py comm ablation    # Part 2 only (Klazy + compression)
    python main.py ada              # adaptive mechanism study
    python main.py scale            # dimension scalability study

OUTPUT
------
    results/            regular benchmark
    results_robust/     robustness study
    results_comm/       communication cost study
    results_ada/        adaptive mechanism study
    results_scale/      dimension scalability study
"""

import sys
import os
import shutil
import numpy as np

_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

np.random.seed(42)

# Result directories for each mode
_RESULT_DIRS = {
    "regular": os.path.join(_root, "results", "main"),
    "robust":  os.path.join(_root, "results", "robust"),
    "comm":    os.path.join(_root, "results", "comm"),
    "ada":     os.path.join(_root, "results", "ada"),
    "scale":   os.path.join(_root, "results", "scale"),
}


def _banner(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def _clear_results(mode: str) -> None:
    """Remove and recreate the results directory for the given mode."""
    d = _RESULT_DIRS.get(mode)
    if d and os.path.exists(d):
        shutil.rmtree(d)
        print(f"[clean] Removed old results: {d}")
    if d:
        os.makedirs(d, exist_ok=True)


def _clear_all() -> None:
    """Remove all result directories."""
    for mode in _RESULT_DIRS:
        _clear_results(mode)


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    if mode == "regular":
        _clear_results("regular")
        func_group = sys.argv[2] if len(sys.argv) > 2 else "all"
        from experiments.benchmarks.run_regular import run_regular
        run_regular(func_group)

    elif mode == "robust":
        _clear_results("robust")
        part = sys.argv[2] if len(sys.argv) > 2 else "all"
        from experiments.ablation.run_robust import run_robust
        run_robust(part)

    elif mode == "comm":
        _clear_results("comm")
        part = sys.argv[2] if len(sys.argv) > 2 else "all"
        from experiments.ablation.run_comm import run_comm
        run_comm(part)

    elif mode == "ada":
        _clear_results("ada")
        from experiments.ablation.run_ada import run_ada
        run_ada()

    elif mode == "scale":
        _clear_results("scale")
        from experiments.benchmarks.run_scalability import run_scalability
        run_scalability()

    elif mode == "all":
        _run_all()

    elif mode == "clean":
        _clear_all()
        cache_root = os.path.join(_root, "_run_cache")
        if os.path.exists(cache_root):
            shutil.rmtree(cache_root)
            print(f"[clean] Removed resume cache: {cache_root}")
        print("[clean] All result directories cleared.")

    else:
        print(
            f"Unknown mode '{mode}'.\n"
            "Available: regular | robust | comm | ada | scale | all | clean"
        )
        sys.exit(1)


def _run_all() -> None:
    """Run all 4 experiments sequentially with fresh result directories.

    Order: regular → comm → ada → robust (robust last because 100 MC is slow).
    """
    _clear_all()

    _banner("Step 1/4 - Regular benchmark (9 functions, d=30, 20 MC)")
    from experiments.benchmarks.run_regular import run_regular
    run_regular("all")

    _banner("Step 2/4 - Communication cost study (Ce benefit + ablation)")
    from experiments.ablation.run_comm import run_comm
    run_comm("all")

    _banner("Step 3/4 - Adaptive mechanism study (M trajectory + comparisons)")
    from experiments.ablation.run_ada import run_ada
    run_ada()

    _banner("Step 4/4 - Robustness study (100 MC + param sweep)")
    from experiments.ablation.run_robust import run_robust
    run_robust("all")

    _banner("All experiments complete.")
    print(
        "\nOutput directories:\n"
        "  results/            regular benchmark\n"
        "  results_comm/       communication cost study\n"
        "  results_ada/        adaptive mechanism study\n"
        "  results_robust/     robustness study\n"
    )


if __name__ == "__main__":
    main()
