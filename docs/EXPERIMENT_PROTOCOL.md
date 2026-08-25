# STDE-CDM reported experiment protocol

The machine-readable frozen settings are under `configs/`. This document states
the scientific rules that connect those configurations.

## Data and split

1. Use `data/wind_data_all_zone.csv` without modifying the target values.
2. Jointly model Zones 1 through 5 as one `24 x 5` power trajectory conditioned
   on a `24 x 5 x 4` NWP tensor.
3. Use the fixed random-state-0 split: 631 LS days, 50 VS days, and 50 locked
   TEST days.
4. Estimate normalization statistics on LS only.
5. Use VS only for checkpoint selection, fusion-weight selection, and any
   hyperparameter decision.
6. Do not use TEST or FICA results to select the forecasting method.

The exact data settings are frozen in `configs/paper_data_split.json`.

## Training

Train Joint WGAN GP, Joint VAE, Joint UMNN, Joint DDPM, Joint CLDM, and the
STDE-CDM spatiotemporal expert with training seeds 0 through 9. Every method
uses the same LS, VS, TEST dates and the same five-site forecasting target.

STDE-CDM combines the Joint CLDM distribution-expert trajectory and the
spatiotemporal-expert trajectory with weights 0.6 and 0.4. The weight is fixed
before locked TEST evaluation. Its two reverse processes use matched sampling
noise.

Architectures, optimizers, epoch counts, and checkpoint patterns are frozen in
`configs/paper_model_training.json`.

## Locked TEST evaluation

For each of ten training seeds, generate 200 joint scenarios on each of the
same 50 locked TEST days. Report MAE, RMSE, CRPS, PS, ES, and VS; lower is
better for every metric. First average over the 50 TEST days for each seed,
then report the mean and sample standard deviation across the ten seeds.

Compare STDE-CDM with Joint CLDM using one-sided paired t tests and Wilcoxon
signed-rank tests over matched training seeds. Construct the reported 95%
confidence intervals by paired bootstrap resampling of the ten matched seeds
with 10,000 resamples.

The exact sampling seeds and statistical settings are frozen in
`configs/paper_locked_test_evaluation.json`.

## FICA downstream validation

Treat FICA as an external application experiment rather than a model-selection
stage. Use the seed-0 Joint CLDM and STDE-CDM checkpoints, each method's native
deterministic forecast head, and the same IEEE RTS 24 physical-system settings.

For every locked TEST day, generate a 6,000-scenario candidate pool. Form the
reported 200-scenario optimization set from at most 100 constraint-informative
scenarios and a uniformly sampled remainder. Evaluate the optimized policy on
an independent 5,000-scenario pool and on the observed TEST trajectory. The
observation is never used to construct or select the optimization scenarios.

The full solver, seed, tolerance, scenario-selection, and restart settings are
frozen in `configs/paper_fica_backtest.json`.

## Reporting and provenance

Retain the effective configuration, training seed, data dates, validation
checkpoint, locked TEST metrics, runtime, and artifact paths. Final
checkpoints are under `artifacts/checkpoints/`; frozen metric records are under
`results/metrics/`; compact FICA results are under `results/fica/`.
