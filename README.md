# DisGRem

Reference implementation for the Distributed Gradient-Regularized Newton
Method (DisGRem) and its communication-efficient and adaptive variants.

**Status:** published  
**Paper:** [arXiv:2605.19396](https://arxiv.org/abs/2605.19396)  
**License:** MIT

## Overview

DisGRem is a decentralized second-order method for consensus optimization over
networks. Each agent solves a local regularized Newton system with vanishing
gradient-norm regularization and communicates through scheduled gossip mixing.
The implementation includes the full DisGRem method, the communication-efficient
CeDisGRem variant, and the adaptive AdaDisGRem variant used in the paper.

## Repository Structure

```text
codes/
  main.py                  unified command-line entry point
  solvers/
    disgrem/               proposed DisGRem-family methods
    baselines/             comparison algorithms
  problems/                benchmark objectives and data interfaces
  experiments/
    benchmarks/            main benchmark and scalability experiments
    ablation/              robustness, communication, and adaptive studies
  scripts/                 figure and table regeneration helpers
  utils/                   graph generation, logging, plotting, and exports
  tests/                   smoke and diagnostic tests
```

Generated outputs are written under `codes/results/` and are not part of the
source release.

## Setup

Requires Python 3.10 or newer.

```bash
cd codes
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\activate
```

## Running Experiments

All commands below are run from `codes/`.

```bash
python main.py regular
python main.py robust
python main.py comm
python main.py ada
python main.py scale
```

To run the full suite used in the paper:

```bash
python main.py all
python main.py scale
python scripts/replot.py
```

The full benchmark can take several hours on a modern multi-core CPU. Individual
experiment modes can be run independently.

## Algorithms

| Code | Paper name | Role |
|------|------------|------|
| `solvers/disgrem/ce_disgrem.py` | DisGRem | Full Hessian exchange |
| `solvers/disgrem/ce_ada_disgrem.py` | AdaDisGRem | Adaptive regularization |
| `solvers/disgrem/disgrem.py` | DisGRem wrapper | Convenience wrapper |
| `solvers/disgrem/ada_disgrem.py` | AdaDisGRem wrapper | Convenience wrapper |
| `solvers/disgrem/dis_greqm.py` | DisGreQm | Quasi-Newton variant |

Baselines include EXTRA, DIGing, DQM, ESOM, SONATA, NetworkGIANT, and DisQN.

## Citation

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

This project is released under the MIT License. See `LICENSE`.
