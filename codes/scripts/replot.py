"""
replot.py  -  Regenerate figures from cached data (no experiments run).

Usage:  python3 replot.py
"""
import os, sys, pickle, csv


class _NumpyCompat(pickle.Unpickler):
    """Handle numpy 2.x pickles on numpy 1.x."""
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from utils.alg.alg_bank import get_alg_bank
from utils.export.plot_utils import (
    fig_plot_multiobj_custom,
    fig_plot_multiobj_repr,
    fig_perf_profiles_tol_panel,
    fig_perf_profiles_comm_panel,
    fig_ce_benefit,
)


def replot_regular():
    """Reload pickle cache and regenerate all regular figures."""
    cache_dir = os.path.join(_root, "_run_cache", "regular")
    results_dir = os.path.join(_root, "results", "main")

    if not os.path.isdir(cache_dir):
        print("[skip] No regular cache found.")
        return

    all_logs = {}
    all_individual = {}
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.endswith(".pkl"):
            continue
        obj_name = fname[:-4]
        with open(os.path.join(cache_dir, fname), "rb") as fh:
            payload = _NumpyCompat(fh).load()
        if (isinstance(payload, dict)
                and "merged" in payload
                and "individual" in payload):
            all_logs[obj_name] = payload["merged"]
            all_individual[obj_name] = payload["individual"]
        else:
            all_logs[obj_name] = payload
        print(f"  Loaded {obj_name}")

    if not all_logs:
        print("[skip] No cached logs.")
        return

    alg_bank = get_alg_bank("MainComp")
    print(f"\n[replot] {len(all_logs)} functions, {len(alg_bank)} algorithms")

    for x_key, y_key in [
        ("steps", "relF"),
        ("steps", "combo"),
        ("timeCost", "relF"),
        ("timeCost", "combo"),
        ("commCost", "relF"),
        ("commCost", "combo"),
    ]:
        print(f"  Plotting multiobj {x_key} vs {y_key} ...")
        fig_plot_multiobj_custom(all_logs, alg_bank, x_key, y_key,
                                 "semilogy", results_dir, n_cols=3)

    print("  Plotting repr_steps_vs_relF (2x2 main-text figure) ...")
    fig_plot_multiobj_repr(all_logs, alg_bank, "steps", "relF",
                           "semilogy", results_dir)

    print("  Plotting performance profiles (iteration) ...")
    fig_perf_profiles_tol_panel(all_logs, alg_bank, results_dir,
                                tol_levels=[1e-3, 1e-6, 1e-9],
                                all_individual=all_individual)

    print("  Plotting performance profiles (comm) ...")
    fig_perf_profiles_comm_panel(all_logs, alg_bank, results_dir,
                                 tol_levels=[1e-3, 1e-6, 1e-9],
                                 all_individual=all_individual)

    print("[done] Regular figures regenerated.\n")


def replot_ce_benefit():
    """Reload ce_benefit.csv and regenerate the ce_benefit figure."""
    results_dir = os.path.join(_root, "results", "comm")
    csv_path = os.path.join(results_dir, "data_log", "ce_benefit.csv")
    if not os.path.isfile(csv_path):
        print("[skip] No ce_benefit.csv found.")
        return

    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        tol_levels = []
        for col in header[2:]:
            tol_levels.append(float(col.split("=")[1]))

        all_data = {}
        for row in reader:
            obj_name = row[0]
            alg_name = row[1]
            vals = []
            for v in row[2:]:
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(float("nan"))
            if obj_name not in all_data:
                all_data[obj_name] = {}
            all_data[obj_name][alg_name] = vals

    ce_pairs = [
        ("DisGrem", "CeDisGrem"),
        ("AdaDisGrem", "CeAdaDisGrem"),
    ]

    print(f"[replot] ce_benefit: {len(all_data)} functions, {len(tol_levels)} tol levels")
    fig_ce_benefit(all_data, ce_pairs, tol_levels, results_dir)
    print("[done] ce_benefit figure regenerated.\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  Regenerating figures from cached data")
    print("=" * 60)
    replot_regular()
    replot_ce_benefit()
    print("All done.")
