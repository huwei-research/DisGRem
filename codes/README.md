# DisGRem Code

See the [project README](../README.md) for full documentation, installation, and usage instructions.

## Quick Reference

```bash
cd codes
python main.py [mode] [arg]
```

| Mode | Description |
|------|-------------|
| `regular` | Main 9-function benchmark (d=30, 20 MC) |
| `robust` | Robustness study (100 MC + param sweep) |
| `comm` | Communication cost study |
| `ada` | Adaptive mechanism study |
| `scale` | Dimension scalability study |
| `all` | Run regular + comm + ada + robust sequentially |
| `clean` | Remove all generated results and caches |
