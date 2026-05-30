#!/usr/bin/env bash
#
# rerun_all.sh — Re-run ALL experiments and regenerate ALL figures.
#
# Usage (from project root):
#   bash codes/scripts/rerun_all.sh           # run everything
#   bash codes/scripts/rerun_all.sh quick     # skip robust/scalability (faster)
#
set -euo pipefail
cd "$(dirname "$0")/.."   # codes/ (parent of scripts/)

echo "============================================================"
echo "  DisGrem Full Experiment Pipeline"
echo "  Started: $(date)"
echo "============================================================"

MODE="${1:-full}"

# 1. Regular benchmark (convergence + profiles + repr figure)
echo ""
echo "[1/5] Regular benchmark ..."
python main.py regular all
echo "  => results/main/ updated"

# 2. Communication cost study (Ce benefit + Klazy/compression ablation)
echo ""
echo "[2/5] Communication cost study ..."
python main.py comm all
echo "  => results/comm/ updated"

# 3. Adaptive mechanism study
echo ""
echo "[3/5] Adaptive mechanism study ..."
python main.py ada
echo "  => results/ada/ updated"

if [[ "$MODE" != "quick" ]]; then

# 4. Robustness study (starting-point + parameter sensitivity)
echo ""
echo "[4/5] Robustness study ..."
python main.py robust all
echo "  => results/robust/ updated"

# 5. Scalability study
echo ""
echo "[5/5] Scalability study ..."
python main.py scale
echo "  => results/scale/ updated"

else
echo ""
echo "[4/5] Robustness study — SKIPPED (quick mode)"
echo "[5/5] Scalability study — SKIPPED (quick mode)"
fi

# 6. Regenerate any remaining figures from cache
echo ""
echo "[post] Regenerating cached figures (replot.py) ..."
python scripts/replot.py

# 7. Recompile paper
echo ""
echo "[post] Recompiling LaTeX (3 passes) ..."
cd ../paper
pdflatex -interaction=nonstopmode disgrem_paper_v9_siam.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode disgrem_paper_v9_siam.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode disgrem_paper_v9_siam.tex > /dev/null 2>&1
cd ..

echo ""
echo "============================================================"
echo "  All done!  $(date)"
echo "  PDF: paper/disgrem_paper_v9_siam.pdf"
echo "============================================================"
