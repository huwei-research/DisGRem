"""
collect_figures.py - Collect all paper-ready figures into a flat directory.

Usage:
    python collect_figures.py                    # default: paper_figures/
    python collect_figures.py --out my_figs/     # custom output directory

Features:
  1. Scans all results*/fig_*/ directories for PDF files.
  2. Copies them to a single flat output directory.
  3. Supports an optional rename mapping (dict below) for paper figure numbers.
  4. Generates an index.html gallery for quick visual review.
"""

from __future__ import annotations
import os
import shutil
import argparse
from pathlib import Path

_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Rename mapping ─────────────────────────────────────────────────────────
# Map source basenames (without extension) to paper-friendly names.
# Only entries present here get renamed; unlisted files keep their names.
# Modify this dict to match your paper's figure numbering.
RENAME_MAP = {
    # Main convergence results (Section IV-A)
    "multiobj_steps_vs_combo":    "fig2_convergence_steps_combo",
    "multiobj_commCost_vs_combo": "fig3_convergence_comm_combo",
    "multiobj_steps_vs_relF":     "fig4_convergence_steps_relF",
    "multiobj_commCost_vs_relF":  "fig5_convergence_comm_relF",
    # Performance / data profiles (Section IV-B)
    "perf_profiles_panel":        "fig6_perf_data_profiles",
    # Convergence trio: steps / comm / time (Section IV-C)
    "trio_ridge":                 "fig7a_trio_ridge",
    "trio_logsumexp":             "fig7b_trio_logsumexp",
    # Communication efficiency (Section IV-D)
    "ce_benefit_steps":           "fig8a_ce_benefit_steps",
    "ce_benefit_comm":            "fig8b_ce_benefit_comm",
    "comm_savings":               "fig9_comm_savings",
    # Scalability (Section IV-E)
    "scalability_panel_ridge":    "fig10a_scalability_ridge",
    "scalability_multiobj_panel": "fig10b_scalability_multi",
    # Robustness (Section IV-F)
    "robust_panel":               "fig11_robust_panel",
    "robust_heatmap":             "fig12_robust_heatmap",
    # Topology (Section IV-G)
    "topology_combined":          "fig13_topology",
    # Condition number sensitivity (Appendix)
    "kappa_sweep":                "figA1_kappa_sweep",
    # Data heterogeneity (Appendix)
    "hetero_comparison":          "figA2_hetero_convergence",
    # Ablation (Appendix)
    "ablation_NC_sweep":          "figA3_ablation_NC",
    "ablation_Klazy_sweep":       "figA4_ablation_Klazy",
    "ablation_ada_vs_fixed":      "figA5_ablation_adaptive",
}


def _find_pdfs(root: Path) -> list[Path]:
    """Recursively find all PDF files under results*/ directories.

    Prioritises paper/ over supplement/ via sort order so paper figures
    are listed first in the gallery.
    """
    pdfs = []
    for d in sorted(root.glob("results*")):
        if d.is_dir():
            paper_dir = d / "paper"
            supp_dir = d / "supplement"
            if paper_dir.is_dir():
                pdfs.extend(sorted(paper_dir.rglob("*.pdf")))
            if supp_dir.is_dir():
                pdfs.extend(sorted(supp_dir.rglob("*.pdf")))
            pdfs.extend(sorted(
                p for p in d.rglob("*.pdf")
                if "paper" not in p.parts and "supplement" not in p.parts
            ))
    return pdfs


def _generate_html(out_dir: Path, files: list[tuple[str, str]]) -> None:
    """Generate a simple HTML gallery listing all collected figures."""
    rows = []
    for orig, dest in sorted(files, key=lambda x: x[1]):
        # Use PNG version if available, else embed PDF
        png_name = dest.replace(".pdf", ".png")
        png_path = out_dir / png_name
        if png_path.exists():
            img_tag = f'<img src="{png_name}" style="max-width:100%;">'
        else:
            img_tag = f'<embed src="{dest}" type="application/pdf" width="100%" height="400px">'
        rows.append(
            f'<div class="card">'
            f'<h3>{dest}</h3>'
            f'<p class="orig">Source: {orig}</p>'
            f'{img_tag}'
            f'</div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Paper Figures Gallery</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f8f8f8; }}
h1 {{ color: #333; }}
.card {{ background: #fff; border: 1px solid #ddd; border-radius: 6px;
         padding: 12px; margin: 12px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.card h3 {{ margin: 0 0 4px 0; font-size: 14px; color: #1a73e8; }}
.card .orig {{ font-size: 11px; color: #888; margin: 0 0 8px 0; }}
.card img {{ border: 1px solid #eee; }}
</style>
</head>
<body>
<h1>Paper Figures ({len(files)} files)</h1>
{''.join(rows)}
</body>
</html>"""

    idx_path = out_dir / "index.html"
    idx_path.write_text(html, encoding="utf-8")
    print(f"[Gallery] {idx_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect paper figures.")
    parser.add_argument("--out", default="paper_figures",
                        help="Output directory (default: paper_figures/)")
    args = parser.parse_args()

    out_dir = _ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = _find_pdfs(_ROOT)
    if not pdfs:
        print("[Warning] No PDF files found under results*/.")
        return

    collected = []
    seen_names = set()

    for pdf_path in pdfs:
        stem = pdf_path.stem
        new_stem = RENAME_MAP.get(stem, stem)

        # Avoid name collisions by appending parent dir prefix
        if new_stem in seen_names:
            parent_name = pdf_path.parent.name
            new_stem = f"{parent_name}_{new_stem}"
        seen_names.add(new_stem)

        dest_name = f"{new_stem}.pdf"
        dest_path = out_dir / dest_name
        shutil.copy2(pdf_path, dest_path)

        # Also copy PNG if it exists alongside the PDF
        png_src = pdf_path.with_suffix(".png")
        if png_src.exists():
            shutil.copy2(png_src, out_dir / f"{new_stem}.png")

        rel_src = pdf_path.relative_to(_ROOT)
        collected.append((str(rel_src), dest_name))

    print(f"\n[collect_figures] Collected {len(collected)} PDF files → {out_dir}/")

    _generate_html(out_dir, collected)

    # Summary
    renamed = [(o, d) for o, d in collected if Path(o).stem != Path(d).stem]
    if renamed:
        print(f"\nRenamed files ({len(renamed)}):")
        for orig, dest in renamed:
            print(f"  {Path(orig).name}  →  {dest}")


if __name__ == "__main__":
    main()
