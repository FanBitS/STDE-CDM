# Reproduction configurations

This directory retains only the configurations used for the reported study:

- `paper_data_split.json`: data source, joint tensor construction, locked split,
  and normalization;
- `paper_model_training.json`: STDE-CDM and five joint baseline trainers;
- `paper_locked_test_evaluation.json`: ten seeds, 50 locked TEST days,
  200 scenarios per day, and statistical analysis;
- `paper_fica_backtest.json`: restart-safe downstream FICA validation.

The `paper_` filename prefix identifies the frozen reported configuration; it
does not imply that manuscript files are included in this repository. Each JSON
record names its maintained Python entry point and effective settings.
