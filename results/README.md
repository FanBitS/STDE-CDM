# Frozen results

- `metrics/` contains the reported scenario-generation metrics, per-seed
  records, calibration result, and ablation summary.
- `fica/` contains the formal 50-day aggregate dispatch records and the compact
  day-00 STDE-CDM case used by the FICA figure.

Large per-day candidate and independent-validation pools are regenerable and
are intentionally excluded from the release. Use the maintained scripts under
`scripts/` to recreate them.
