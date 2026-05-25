# DisGRem: Distributed Gradient-Regularized Newton Method

Reference implementation of the **DisGRem** algorithm family for decentralized second-order consensus optimization over networks.

> **Paper:** W. Hu, P. Xie, Y.-X. Yuan, and L. Zhang,
> *Distributed Gradient-Regularized Newton Method: Scheduled Consensus and O(epsilon^{-1}) Global Iteration Complexity*,
> [arXiv:2605.19396](https://arxiv.org/abs/2605.19396), 2026.

## Highlights

- **DisGRem** — each agent solves a local regularized Newton system with vanishing gradient-norm regularization and communicates via two-stage gossip mixing.
- **CeDisGRem** — communication-efficient variant using top-k compressed Hessian tracking with lazy updates.
- **AdaDisGRem** — adaptive regularization parameter M that self-tunes during optimization.
- Post-burn-in, achieves the centralized regularized Newton rate O(epsilon^{-1}) **without line search or stepsize tuning**.
- Under strong convexity, superlinear convergence of order 3/2.

## Repository Structure

```
codes/
├── main.py                 # unified CLI entry point
├── solvers/
│   ├── disgrem/            # proposed methods (5 variants)
│   └── baselines/          # 7 comparison algorithms
├── problems/               # 9 test objective functions
├── experiments/
│   ├── benchmarks/         # main benchmark + scalability
│   └── ablation/           # robustness, comm cost, adaptive M
├── utils/                  # graph generation, plotting, logging
├── scripts/                # figure regeneration, table export
├── data/                   # bundled datasets (svmguide3, a9a)
├── tests/                  # diagnostic scripts
└── results/                # experiment output (generated)
```

### Algorithms

| Code | Paper name | Key feature |
|------|-----------|-------------|
| `ce_disgrem.py` | DisGRem | Full Hessian exchange |
| `ce_ada_disgrem.py` | AdaDisGRem | Adaptive M |
| `disgrem.py` / `ada_disgrem.py` | (wrappers) | Thin wrappers |
| `dis_greqm.py` | DisGreQm | Quasi-Newton variant |

Baselines: EXTRA, DIGing, DQM, ESOM, SONATA, NetworkGIANT, DisQN.

## Installation

Requires **Python >= 3.10**.

```bash
git clone https://github.com/huwei0121/DisGRem.git
cd DisGRem/codes
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Quick Start

All experiments are launched from the `codes/` directory:

```bash
cd codes

# Run a single experiment
python main.py regular          # main 9-function benchmark (d=30, 20 MC)
python main.py robust           # robustness study (100 MC + param sweep)
python main.py comm             # communication cost study
python main.py ada              # adaptive mechanism study
python main.py scale            # dimension scalability study

# Run all experiments sequentially
python main.py all

# Sub-modes (examples)
python main.py regular ridge    # benchmark on ridge regression only
python main.py robust start     # robustness Part 1 only
python main.py comm ce          # communication efficiency Part 1 only

# Clean all generated results
python main.py clean
```

### Output

Results are saved under `codes/results/`:

| Mode | Directory | Contents |
|------|-----------|----------|
| `regular` | `results/main/` | Per-function summary `.txt` files |
| `robust` | `results/robust/` | Success rate CSV + data logs |
| `comm` | `results/comm/` | CE benefit CSV + savings tables |
| `ada` | `results/ada/` | M-trajectory data |
| `scale` | `results/scale/` | Scalability data |

### Regenerating Figures

After running experiments, regenerate paper figures from saved data:

```bash
cd codes
python scripts/replot.py
```

## Reproducing Paper Results

To reproduce the full experiment suite from the paper (Section 6):

```bash
cd codes
python main.py all        # runs regular, comm, ada, robust (~several hours)
python main.py scale      # scalability study (separate)
python scripts/replot.py  # regenerate all figures
```

**Hardware note:** The full benchmark runs 20 Monte Carlo trials per function with 10 algorithms; `robust` runs 100 MC trials. Expect several hours on a modern multi-core CPU. Individual experiments can be run selectively.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{hu2026disgrem,
  title   = {Distributed Gradient-Regularized Newton Method:
             Scheduled Consensus and {$\mathcal{O}(\varepsilon^{-1})$}
             Global Iteration Complexity},
  author  = {Hu, Wei and Xie, Pengcheng and Yuan, Ya-Xiang and Zhang, Li},
  journal = {arXiv preprint arXiv:2605.19396},
  year    = {2026}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

Wei Hu — [huwei@amss.ac.cn](mailto:huwei@amss.ac.cn)
