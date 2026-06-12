# Public DisGRem Agent Instructions

## Project Identity

This is the public MIT-licensed implementation of DisGRem, CeDisGRem, and
AdaDisGRem for decentralized consensus optimization. It is tied to the paper
but should remain a clean public code release.

## Key Paths

- Command-line entry: `codes/main.py`.
- Proposed methods: `codes/solvers/disgrem/`.
- Baselines: `codes/solvers/baselines/`.
- Problems and data interfaces: `codes/problems/`.
- Benchmarks: `codes/experiments/benchmarks/`.
- Ablations: `codes/experiments/ablation/`.
- Tests: `codes/tests/`.

## Public Release Rules

- Do not add private manuscript drafts, review material, or unpublished result
  dumps.
- Preserve the distinction between DisGRem, communication-efficient variants,
  adaptive variants, and baselines.
- Keep generated outputs under `codes/results/` out of commits unless the user
  explicitly promotes them for release.
- Maintain clean setup and test commands for outside users.

## Verification

```powershell
cd codes
python -m pytest tests
python main.py regular
```

Use targeted benchmark or ablation scripts when changing solver behavior,
communication schedules, graph generation, or adaptive regularization.

## Research Rules

- Claims about convergence, communication efficiency, and scalability must be
  traceable to the paper or reproducible experiment scripts.
- Do not weaken graph, smoothness, or consensus assumptions in prose or comments.
