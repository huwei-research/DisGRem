# Project Conventions

This document defines the naming, formatting, and structural conventions for all research
projects under `2026Projects/`. Every new project and every modification to an existing
project should follow these rules.

---

## 1. Project Structure

Every project at the top level must contain:

```
ProjectX/
├── README.md              # Project overview (title, abstract, structure index)
├── paper/                 # LaTeX sources and paper figures
│   ├── {project}_paper_v{M}_{m}.tex   # Main paper (M=major, m=minor)
│   ├── {project}_supplementary.tex    # Supplementary material (optional)
│   ├── {project}_paper_v{M}_{m}_shared.tex  # Shared macros (optional)
│   ├── {project}_refs.bib             # Bibliography
│   ├── figures/           # Publication-quality figures (tracked by Git)
│   └── *.cls, *.bst       # Journal style files (keep original names)
├── codes/                 # All runnable code
│   ├── README.md          # Code documentation and Quick Start
│   ├── requirements.txt   # Python dependencies with pinned versions
│   ├── LICENSE
│   ├── .gitignore
│   ├── core/              # Base classes (Solver, Problem, Result, etc.)
│   ├── solvers/           # Solver implementations (one subfolder per algorithm)
│   │   ├── __init__.py    # Unified registry / re-exports
│   │   ├── CATALOG.md     # Algorithm catalog: name, version, one-line description
│   │   ├── <algo_a>/      # Each algorithm gets its own package
│   │   ├── <algo_b>/
│   │   └── external/      # Third-party solver wrappers
│   ├── problems/          # Test problem definitions + helpers
│   ├── experiments/       # Experiment scripts, categorised into subfolders
│   │   ├── benchmarks/    # Formal benchmark scripts
│   │   ├── ablation/      # Ablation studies
│   │   ├── analysis/      # Analysis scripts (convergence, sensitivity, etc.)
│   │   ├── research/      # Exploratory / version-testing scripts
│   │   └── experiment_runner.py  # Shared runner framework (if applicable)
│   ├── tests/             # Unit and integration tests
│   ├── utils/             # Shared utilities (metrics, plotting, I/O, logging)
│   ├── scripts/           # One-off helper scripts (merge data, batch run, replot)
│   ├── results/           # All experiment outputs (CSV, JSON, figures, reports)
│   └── plans/             # Historical experiment plans and checklists
├── presentation/          # Slides and talk materials (optional)
│   ├── {project}_beamer_v{M}_{m}.tex  # Beamer slides
│   ├── {project}_refs.bib             # Bibliography for slides
│   ├── figures/           # Presentation figures (tracked by Git)
│   └── data/              # Data tables used in slides (optional)
└── .gitignore             # Project-level gitignore
```

Optional top-level folders: `data/` (for large static datasets).

---

## 2. Directory Naming

| Rule | Example |
|------|---------|
| Always `lowercase_snake_case` | `bup_newuoa/`, `dfo_tr/` |
| Solver subdirectories: algorithm name | `bup_tr/`, `disgrem/`, `dfo_etr/` |
| Experiment categories: fixed set | `benchmarks/`, `ablation/`, `analysis/`, `research/` |
| Result subdirectories: experiment name | `noiseless/`, `v7_ablation/`, `track_ab_diag/` |

---

## 3. File Naming

### 3.1 Python Files

| Category | Pattern | Example |
|----------|---------|---------|
| Solver (main) | `{algorithm}.py` | `bup_tr.py`, `disgrem.py` |
| Solver (version) | `{algorithm}_v{N}.py` | `bup_newuoa_v7.py` |
| External wrapper | `{library}_wrapper.py` or `external_solvers.py` | `pdfo_wrapper.py` |
| Test file | `test_{module}.py` | `test_bup_tr.py` |
| Experiment script | `{descriptive_name}.py` | `standard_benchmark.py` |
| Exploratory script | `_{descriptive_name}.py` (underscore prefix) | `_test_v7_quick.py` |
| Helper script | `{verb}_{noun}.py` | `merge_results.py` |

### 3.2 LaTeX Files

| Category | Pattern | Example |
|----------|---------|---------|
| Paper version | `{project}_paper_v{major}_{minor}.tex` | `buptr_paper_v5_0.tex` |
| Supplementary | `{project}_supplementary.tex` | `buptr_supplementary.tex` |
| Shared macros | `{project}_paper_v{M}_{m}_shared.tex` | `disgrem_paper_v10_0_shared.tex` |
| Bibliography | `{project}_refs.bib` | `buptr_refs.bib` |
| Style files | Keep original name | `siamart171218.cls` |
| Beamer slides | `{project}_beamer_v{major}_{minor}.tex` | `buptr_beamer_v1_0.tex` |
| Beamer extras | `{project}_qa_notes.tex` (optional) | `buptr_qa_notes.tex` |

### 3.2.1 Paper Directory Structure

```
paper/
├── {project}_paper_v{M}_{m}.tex        # Current working version
├── {project}_paper_v{M-1}_{m}.tex      # Previous versions (keep for reference)
├── {project}_supplementary.tex          # Optional
├── {project}_paper_v{M}_{m}_shared.tex  # Shared macros / commands (optional)
├── {project}_refs.bib                   # Bibliography database
├── figures/                             # Publication figures (tracked by Git)
│   ├── perf_profile.pdf
│   └── convergence.pdf
├── siamart171218.cls                    # Journal style (keep original name)
└── siamplain.bst                        # Bibliography style (keep original name)
```

- **All LaTeX build artifacts** (`.aux`, `.log`, `.out`, `.thm`, `.bbl`, `.blg`,
  `.synctex.gz`, `.fls`, `.fdb_latexmk`, `texput.log`) must be gitignored and never
  committed. They are regenerated by `pdflatex` / `latexmk`.
- **Previous paper versions** are kept in the same directory for reference. Do not delete
  old `.tex` files — they serve as a historical record.
- The `figures/` subfolder inside `paper/` is tracked by Git (excluded from the global
  figure-ignore rules).

### 3.2.2 Presentation Directory Structure

```
presentation/
├── {project}_beamer_v{M}_{m}.tex       # Beamer slides
├── {project}_qa_notes.tex              # Q&A prep notes (optional)
├── {project}_refs.bib                  # Bibliography for slides
├── figures/                            # Presentation figures (tracked by Git)
│   ├── perf_profile.pdf
│   └── trajectory.pdf
└── data/                               # Data tables embedded in slides (optional)
    └── summary.csv
```

- Beamer build artifacts (`.aux`, `.log`, `.nav`, `.snm`, `.toc`, `.out`, `.vrb`,
  `.bcf`, `.run.xml`, `.bbl`, `.blg`, `.fls`, `.fdb_latexmk`) are gitignored.
- The `figures/` subfolder inside `presentation/` is tracked by Git.
- Do **not** store raw experiment data in `presentation/`. If slides reference experiment
  results, either use a symlink to `codes/results/` or copy only the small summary CSVs
  into `presentation/data/`.

### 3.3 Results and Data Files

| Category | Pattern | Example |
|----------|---------|---------|
| Raw CSV | `{experiment}_{YYYYMMDD_HHMMSS}.csv` | `experiment_20260317_071542.csv` |
| Summary CSV | `{descriptive_name}.csv` | `merged_noiseless.csv` |
| JSON results | `results.json` (one per experiment dir) | `v7_ablation/results.json` |
| Report | `REPORT.md` (uppercase) | `v7_ablation/REPORT.md` |
| Figure (publication) | `{name}.pdf` | `ablation_heatmap.pdf` |
| Figure (quick view) | `{name}.png` | `convergence_grid.png` |

---

## 4. Python Code Style

### 4.1 General

- **PEP 8** compliance.
- **Line length**: 100 characters max.
- **Indentation**: 4 spaces (no tabs).
- **Trailing whitespace**: none.
- **File ending**: single newline at EOF.

### 4.2 Naming

| Entity | Convention | Example |
|--------|-----------|---------|
| Module | `lower_snake_case` | `gp_utils.py` |
| Class | `PascalCase` | `BUPTRSolver` |
| Function / method | `lower_snake_case` | `solve_subproblem()` |
| Variable | `lower_snake_case` | `trust_radius` |
| Constant | `UPPER_SNAKE_CASE` | `MAX_ITERATIONS` |
| Private | Leading underscore | `_fit_gp()` |

### 4.3 Imports

Organize imports in three blocks separated by blank lines:

```python
import os
import sys

import numpy as np
import scipy.linalg as la

from core.base import Solver, OptimizationProblem
from solvers.bup_tr import BUPTRSolver
```

### 4.4 Docstrings

Use **NumPy-style** docstrings for all public functions and classes:

```python
def solve(self, problem, x0, options=None):
    """Solve an unconstrained optimization problem.

    Parameters
    ----------
    problem : OptimizationProblem
        The problem to solve.
    x0 : np.ndarray
        Initial point.
    options : dict, optional
        Solver options.

    Returns
    -------
    OptimizationResult
        The optimisation result containing x_best, f_best, nfev, etc.
    """
```

### 4.5 Comments

- Only explain non-obvious intent, trade-offs, or constraints.
- Do **not** narrate what the code does line-by-line.
- Use `# TODO:` for planned changes; `# HACK:` for workarounds; `# NOTE:` for important context.

---

## 5. Cross-Platform Compatibility

All code must run identically on Windows, Linux, and macOS under Cursor or any Python environment.

### 5.1 Path Handling

- **Always use `pathlib.Path`** for filesystem operations. Never hardcode `\` or `/`.
- For `sys.path` setup in experiment scripts, use this canonical pattern:

```python
import sys
from pathlib import Path

_CODES_ROOT = Path(__file__).resolve().parents[N]  # N = depth from codes/
if str(_CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODES_ROOT))
```

where `N` depends on file location:

| File location | N |
|---------------|---|
| `codes/*.py` | 0 |
| `codes/experiments/*.py` | 1 |
| `codes/experiments/benchmarks/*.py` | 2 |
| `codes/tests/*.py` | 1 |

- Result/output directories must be created with `Path.mkdir(parents=True, exist_ok=True)`.
- Never use `os.path.join` in new code; prefer `Path(...)  / "sub" / "dir"`.
- String paths passed to libraries (matplotlib, pandas) should use `str(path)` explicitly.

### 5.2 Line Endings

Each project **must** include a `.gitattributes` file at the project root:

```gitattributes
# Force LF for all text files (consistent across Windows/Linux/macOS)
* text=auto eol=lf

# Explicitly mark binary files
*.pdf  binary
*.png  binary
*.jpg  binary
*.gif  binary
*.xlsx binary
*.xls  binary
*.gz   binary
*.zip  binary
*.7z   binary

# Shell scripts must be LF
*.sh   text eol=lf
```

### 5.3 Shell Scripts

- Shell scripts (`.sh`) are for reference/convenience only; they are **not** required to run experiments.
- Every experiment must be runnable as `python -m experiments.benchmarks.run_xxx` or `python experiments/benchmarks/run_xxx.py` from the `codes/` directory.
- If platform-specific helper scripts are needed, provide both `.sh` (Linux/macOS) and `.ps1` (Windows) versions, or use Python scripts instead.

### 5.4 Environment Isolation

- Each project uses its own virtual environment inside `codes/`:
  - Linux/macOS: `python -m venv .venv && source .venv/bin/activate`
  - Windows: `python -m venv .venv && .venv\Scripts\activate`
- The `.venv*/` directory is always gitignored.
- `requirements.txt` pins exact versions (`numpy==1.26.4`, not `numpy>=1.26`).

---

## 6. Git Conventions

### 6.1 `.gitignore`

Track numerical data (CSV, JSON, XLS). **Do not track generated figures** (PDF, PNG) -- they
can be regenerated locally by running the plotting code.

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/

# Virtual environments
.venv*/
venv/
env/

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
desktop.ini

# Logs
*.log

# Generated figures (regenerate locally with plotting scripts)
*.pdf
*.png
*.jpg
*.eps
*.svg
# Exception: paper/ figures are tracked (they go through LaTeX)
!paper/**/*.pdf
!paper/**/*.png

# LaTeX build artifacts (paper + beamer)
*.aux
*.out
*.thm
*.bbl
*.blg
*.synctex.gz
*.fls
*.fdb_latexmk
*.run.xml
*.bcf
texput.log

# Beamer-specific build artifacts
*.nav
*.snm
*.toc
*.vrb
```

**What IS tracked:** `.py`, `.csv`, `.json`, `.xls`, `.xlsx`, `.md`, `.tex`, `.txt`, `.sh`,
`.cls`, `.bst`, `.bib`, `requirements.txt`, `LICENSE`.

**What is NOT tracked:** `.pdf` / `.png` (generated figures in `results/`), `.log`,
`__pycache__/`, `.venv*/`, IDE files, OS files, LaTeX build artifacts.

**Rationale:** Figures are deterministic outputs of data + plotting code. Tracking them bloats
the repository with binary diffs. To view figures, run the plotting script locally:

```bash
cd codes
python scripts/replot_from_csv.py        # regenerate all figures
python experiments/analysis/plot_xxx.py   # regenerate specific figure
```

### 6.2 `.gitattributes`

Every project root must have a `.gitattributes` file (see Section 5.2) to ensure consistent
line endings across platforms.

### 6.3 Commit Messages

- Use imperative mood: "Add noisy benchmark results", not "Added..." or "Adds..."
- First line: concise summary (<=72 chars).
- Optional body after blank line for detailed explanation.

### 6.4 Repository Structure

- One Git repository per project (not one for the entire workspace).
- The `.gitignore` lives at the project root **and** inside `codes/`.

---

## 7. Numerical Experiment Conventions

This section codifies the style, structure, and spirit of our numerical experiments.
Following these conventions ensures reproducibility, comparability, and scientific rigor.

### 7.1 Experiment Script Structure

Every experiment script follows this skeleton:

```python
#!/usr/bin/env python
"""One-line title.

Extended description: what is being tested, why, and key hypotheses.

Usage:
    python experiments/benchmarks/xxx.py [--option value]
"""

import sys
from pathlib import Path

_CODES_ROOT = Path(__file__).resolve().parents[2]
if str(_CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODES_ROOT))

import numpy as np

from problems.xxx import get_problem
from solvers import get_solver

# ── Configuration ──────────────────────────────────────────────────────
SOLVERS = ["solver_a", "solver_b"]
PROBLEMS = {"rosenbrock": [2, 10], "sphere": [5, 10]}
SEEDS = [42, 123, 7, 256, 999]
OPTS = {"max_eval": 3000, "rhobeg": 1.0, "rhoend": 1e-8}

RESULTS_DIR = _CODES_ROOT / "results" / "experiment_name"

# ── Main loop ──────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for problem_name, dims in PROBLEMS.items():
        for dim in dims:
            for solver_name in SOLVERS:
                for seed in SEEDS:
                    # ... run and collect results ...
                    pass

    # Save raw data
    df.to_csv(RESULTS_DIR / "raw_results.csv", index=False)

    # Generate figures (optional, can be done separately)
    plot_performance_profiles(df, RESULTS_DIR)

if __name__ == "__main__":
    main()
```

### 7.2 Reproducibility Requirements

- **Fixed seeds**: Every stochastic element uses explicit seeds from the `SEEDS` list.
  Default: `SEEDS = [42, 123, 7, 256, 999]` (5 seeds for statistical significance).
- **Deterministic threading**: Experiment scripts should set single-thread BLAS at the top:

```python
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
```

- **Solver options**: Use a shared `OPTS` dict. Standard defaults:
  - `max_eval`: 3000 (for n<=20 DFO problems)
  - `rhobeg`: 1.0 (initial trust region radius)
  - `rhoend`: 1e-8 (convergence threshold)
- **Problem configurations**: Use `problem_configs.py` for per-problem defaults (x0, rhobeg,
  rhoend overrides). The experiment runner should merge these with global `OPTS`.

### 7.3 Success Metric

All DFO benchmarks use the **More-Wild relative error**:

```
f_rel(x) = (f(x) - f*) / (f(x0) - f* + eps)
```

where `f*` is the known optimal value and `eps` is a small constant (typically 1e-10).
Success at tolerance `tau` means `f_rel(x) < tau`.

Standard tolerance levels for performance profiles: `tau in {1e-1, 1e-3, 1e-5, 1e-7}`.

### 7.4 Result Organization

Each experiment produces a result directory:

```
results/
└── experiment_name/
    ├── raw_results.csv       # Full data: solver, problem, dim, seed, nfev, f_best, ...
    ├── summary.csv           # Aggregated statistics (mean, std, success_rate)
    └── REPORT.md             # Key findings (written by human or AI after analysis)
```

- Figures (`.pdf`, `.png`) are generated into the same directory but **not committed to Git**.
- The plotting code lives in `experiments/` or `scripts/` and reads from the CSV data.

### 7.5 Plotting Conventions

- **Style**: SIAM journal style. Seaborn `whitegrid` palette. Colorblind-friendly colors.
- **Format**: PDF for publication, PNG for quick preview. Both generated by the same function.
- **Font**: LaTeX-compatible (`text.usetex: True` if available, graceful fallback otherwise).
- **Performance profiles**: Use Dolan-More style. X-axis = ratio to best solver. Y-axis = fraction of problems solved.
- **Convergence plots**: X-axis = function evaluations. Y-axis = log10(f_rel).
- **All figure-generating code must be re-runnable**: given only the CSV data and the script, any figure can be regenerated on any machine.

### 7.6 Experiment Categories

| Category | Path | Purpose | Example |
|----------|------|---------|---------|
| Benchmarks | `experiments/benchmarks/` | Formal head-to-head comparisons for paper | `standard_benchmark.py` |
| Ablation | `experiments/ablation/` | Isolate contribution of each component | `_test_v7_full_ablation.py` |
| Analysis | `experiments/analysis/` | Deep dives: convergence, sensitivity, scalability | `sensitivity_analysis.py` |
| Research | `experiments/research/` | Exploratory: new ideas, version testing, quick checks | `_test_v7_quick.py` |

- **Benchmarks** are the gold standard: they define the claims in the paper.
- **Ablation** scripts toggle individual features on/off to measure their contribution.
- **Analysis** scripts explore secondary questions (parameter sensitivity, scaling behaviour).
- **Research** scripts are scratchpads; prefix with `_` if they are quick one-off tests.

### 7.7 Starting a New Project

When creating a new research project:

1. Copy the standard directory template (Section 1).
2. Implement the Solver interface in `core/base.py`.
3. Register solvers in `solvers/__init__.py`.
4. Define problems in `problems/`.
5. Write the first benchmark in `experiments/benchmarks/`.
6. Results automatically go to `results/`.
7. Commit data (CSV/JSON) but not figures.
8. Write `REPORT.md` summarizing findings.

---

## 8. Solver Version Management

- The **main** solver file (e.g., `buptr_newuoa.py`) is the production version.
- Experimental versions use the `_v{N}` suffix and live alongside the main file.
- `CATALOG.md` in `solvers/` documents every algorithm and version with a one-line summary.
- When a versioned solver becomes the new production version, update the main file and CATALOG.md.

---

## 9. Cloning and Syncing on a New Machine

### 9.1 First-Time Setup (Clone)

On a new machine, run the following to get all three projects:

```bash
# Clone all repositories
git clone https://github.com/huwei0121/BUPTR.git
git clone https://github.com/huwei0121/MATRO.git
git clone https://github.com/huwei0121/DisGRem.git
```

Then set up each project's Python environment:

```bash
cd BUPTR/codes
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\activate

pip install -r requirements.txt
```

Repeat for `MATRO` and `DisGRem`.

### 9.2 Daily Sync (Pull / Push)

```bash
# Pull latest changes from GitHub
cd BUPTR
git pull

# After making local changes: stage, commit, and push
git add -A
git commit -m "Describe what changed"
git push
```

### 9.3 Git Configuration for Proxy Environments

If your network requires a proxy (e.g., behind a firewall in China), configure Git:

```bash
# SOCKS5 proxy (common for Clash / V2Ray)
git config --global http.proxy  socks5h://127.0.0.1:7897
git config --global https.proxy socks5h://127.0.0.1:7897

# HTTP proxy
git config --global http.proxy  http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897

# Remove proxy when not needed
git config --global --unset http.proxy
git config --global --unset https.proxy
```

If using GitHub CLI (`gh`) for authentication:

```bash
gh auth login
gh auth setup-git   # sets credential helper for git
```

### 9.4 Resolving Conflicts

- Always `git pull` before starting work.
- If a merge conflict occurs, resolve it manually, then:

```bash
git add <conflicted-file>
git commit -m "Resolve merge conflict in <file>"
git push
```

---

## 10. Regenerating Figures After Clone

Since generated figures (PDF, PNG, JPG, EPS, SVG) are **not tracked by Git** (see Section 6.1),
a freshly cloned repository will not contain any figures in `results/` directories. All the
numerical **data** (CSV, JSON) is tracked, so figures can be fully regenerated from data.

### 10.1 Quick Regeneration

Each project provides plotting scripts that read CSV data and produce figures:

| Project  | Command (run from `codes/`)                            | What it does                      |
|----------|--------------------------------------------------------|-----------------------------------|
| BUPTR    | `python scripts/replot_from_csv.py`                    | Regenerate all result figures      |
| BUPTR    | `python experiments/analysis/replot_profiles.py`       | Regenerate performance profiles    |
| BUPTR    | `python experiments/analysis/plot_convergence_grid.py` | Regenerate convergence grids       |
| DisGRem  | `python scripts/replot.py`                             | Regenerate all result figures      |
| MATRO   | `python utils/plot_trajectory_2d.py`                   | Regenerate trajectory plots        |

### 10.2 General Workflow

After cloning on a new machine:

```bash
cd BUPTR/codes
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Regenerate all figures
python scripts/replot_from_csv.py

# Or regenerate figures for a specific experiment
python experiments/analysis/plot_convergence_grid.py
```

### 10.3 Writing Plotting Scripts

Every experiment that generates figures **must** follow this contract:

1. **Separate data collection from plotting.** The experiment script saves CSV/JSON data.
   The plotting script reads that data and produces figures.
2. **Plotting scripts must be self-contained.** Given only the CSV data and the script, figures
   can be regenerated on any machine with the correct Python environment.
3. **Plotting scripts must accept the data directory as input** (either via command-line argument
   or a clearly defined constant at the top of the file):

```python
RESULTS_DIR = _CODES_ROOT / "results" / "experiment_name"
DATA_FILE   = RESULTS_DIR / "raw_results.csv"
```

4. **Output figures go to the same directory as the data:**

```python
fig.savefig(RESULTS_DIR / "performance_profile.pdf", bbox_inches="tight")
fig.savefig(RESULTS_DIR / "performance_profile.png", dpi=200, bbox_inches="tight")
```

5. **Batch regeneration script** (`scripts/replot_from_csv.py` or `scripts/replot.py`) should
   walk through all `results/` subdirectories and call the appropriate plotting functions.

### 10.4 Paper Figures

Figures in `paper/` **are** tracked by Git (they are excluded from the ignore rules via
`!paper/**/*.pdf` and `!paper/**/*.png`). These are the final publication-quality figures
that are directly included in the LaTeX source.

Workflow for paper figures:
1. Generate figures in `results/` using plotting scripts.
2. Copy the final versions to `paper/figures/` (or use a script like `scripts/collect_figures.py`).
3. Commit the `paper/` figures to Git so the LaTeX document compiles on any machine.
