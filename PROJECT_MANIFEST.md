# STDE-CDM release manifest

This manifest defines the self-contained academic code release. All maintained
entry points resolve paths from the repository root. Manuscript sources,
article PDFs, submission templates, writing notes, and reference-paper files
are intentionally excluded.

## Included components

| Location | Purpose | Status |
|---|---|---|
| `src/stde_cdm/` | STDE-CDM, joint baselines, metrics, data, and FICA interface | Required |
| `src/cldm/` | Independent CLDM implementation and shared data utilities | Required |
| `src/models/` | UMNN normalizing-flow implementation | Required by Joint UMNN |
| `scripts/` | Training, evaluation, visualization, and dispatch entry points | Required |
| `configs/` | Frozen data, training, evaluation, and FICA settings | Required |
| `tests/` | Model, shape, data-interface, and metric checks | Verification |
| `data/` | Aligned multisite wind data and compact reference scenarios | Required |
| `artifacts/checkpoints/` | Final checkpoints for all evaluated seeds | Evaluation without retraining |
| `results/metrics/` | Frozen forecasting metrics and statistical records | Reported results |
| `results/fica/` | Formal 50-day FICA summaries and representative case | Downstream validation |
| `figures/results/` | Reproducible figures and compact plotting data | Visualization |
| `notebooks/` | Inspectable wind-to-FICA trace | Optional |
| `fica_dispatch_optimizer/` | FICA optimizer and retained upstream notice | Fresh dispatch solves |
| `docs/` | Reproduction protocol and verification record | Documentation |

## Frozen experiment inventory

### Data

- `data/wind_data_all_zone.csv` contains 731 complete days per wind zone.
- The joint experiment uses the first five zones.
- Split: 631 learning days, 50 validation days, and 50 locked TEST days with
  random state 0.
- Dataset SHA256:
  `633e85d3a23d278c3e35c287568f07c2ff86590e3be50e22c8b169785a13bcfe`.
- `data/generated/wind_UMNN_M_1_z0-1-2-3-4_d0_n6000.npz` is the compact
  6,000-scenario reference pool used by the trace notebook.

### Checkpoints

- 60 final checkpoints: six evaluated model families × ten training seeds.
- Model families: Joint WGAN GP, Joint VAE, Joint UMNN, Joint DDPM,
  Joint CLDM, and STDE-CDM.
- Five additional sitewise CLDM checkpoints support the packaged calibration
  path.
- No individual tracked file exceeds GitHub's 100 MB file limit.

### Scenario evaluation

- Same 50 locked TEST days for every model.
- 200 synchronized multisite scenarios per day.
- Matched training seeds 0 through 9.
- Metrics: MAE, RMSE, CRPS, PS, ES, and VS; lower is better.
- Authoritative record: `results/metrics/paper_locked_test_10seed.json`.

### FICA validation

- Models: seed-0 Joint CLDM and STDE-CDM.
- 50 locked TEST days.
- Each model uses its native deterministic forecast.
- Candidate distribution: 6,000 generated scenarios per model and day.
- Fixed optimization budget: 200 constraint-relevant scenarios.
- Independent validation: 5,000 generated scenarios plus the locked real
  trajectory.
- Aggregate table: `results/fica/all_case_metrics.csv`.
- Summary: `results/fica/final_summary.json`.
- Representative arrays: `results/fica/day_00/stde_cdm_witness_200.npz`.

## Verification status

- All Python files in `src/`, `scripts/`, and `tests/` compile.
- The automated suite passes 13 tests.
- Every trainer completed a one-epoch smoke run on the packaged data.
- All six model families were loaded and sampled for seeds 0 through 9.
- Quantitative plotting entry points regenerate their outputs under
  `figures/results/`.
- The FICA adapter reached Gurobi optimal status on the IEEE RTS 24 case.

Detailed evidence is retained in `docs/SCRIPT_VERIFICATION.md`.

## Regenerable files not included

- Intermediate checkpoints, optimizer states, caches, and exploratory runs.
- Approximately 1.5 GiB of per-day FICA candidate and validation pools.
- Gurobi binaries and license files.
- Manuscripts, article PDFs, submission layouts, private writing notes, and
  reference-paper collections.

## Path and integrity policy

Maintained Python entry points contain no dependency on the original workspace.
Historical provenance paths in frozen manifests are normalized to repository
relative identifiers. SHA256 values are content fingerprints used to detect
accidental changes; they do not reveal file contents or credentials.

Third-party provenance and licensing are documented in `THIRD_PARTY.md`.
