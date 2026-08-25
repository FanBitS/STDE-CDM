# Reproducible figures

`results/` contains generated PDF and PNG figures together with the compact NPZ
and JSON records consumed by the plotting scripts.

Rebuild the quantitative figures from the repository root:

```bash
python scripts/make_paper_visualizations.py
python scripts/make_locked_test_dm_tests.py
python scripts/make_fica_backtest_figure.py
```

The inputs are the frozen metrics under `results/metrics/`, final checkpoints
under `artifacts/checkpoints/`, wind data under `data/`, and compact FICA case
under `results/fica/`. All outputs remain in `figures/results/`.
