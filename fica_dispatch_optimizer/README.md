# FICA dispatch optimizer

This directory contains the third-party FICA/EIFICA power-system dispatch
solver used for the downstream operational validation of STDE-CDM.
It is not part of the STDE-CDM scenario-generation network.

The maintained STDE-CDM entry points convert generated wind trajectories into
the deterministic forecast and forecast-error arrays expected by this solver.
Gurobi then optimizes day-ahead generator schedules and AGC participation
factors subject to generator, ramping, transmission, power-balance, and
distributionally robust joint chance constraints.

## Files

- `solar_all_method.py`: core FICA, EIFICA, and CVaR optimization routines and
  joint-constraint checker. The upstream filename is retained for traceability;
  the STDE-CDM adapter uses it for wind power.
- `paper_eval.py`: upstream parameter-grid experiment driver. It is retained
  for method provenance and is not the main STDE-CDM 50-day backtest entry.
- `solar_scenario_gen.py`: upstream photovoltaic scenario utility; not used to
  generate the STDE-CDM wind scenarios.
- `test_gurobi_limit.py`: Gurobi license and model-size diagnostic.
- `data/UK_norm_load_curve_highest.npy`: load profile used by the dispatch case.
- `UPSTREAM_README.md` and `UPSTREAM_LICENSE`: original documentation and
  GPL-3.0 license.

The primary project adapter is `src/stde_cdm/fica_system.py`. The formal
50-day study is launched from `scripts/run_fica_native_sthead_50day.py`.
