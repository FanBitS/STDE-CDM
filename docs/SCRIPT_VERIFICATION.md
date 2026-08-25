# Release verification record

Verification date: 2026-07-29
Environment: Conda `dl`, Python 3.11, PyTorch with NVIDIA RTX 5090
Project root: the portable `STDE-CDM` repository

## Static and unit checks

- Every Python file in `src/`, `scripts/`, and `tests/` compiles.
- All command-line scripts return successfully for `--help`.
- The test suite passes: 13 tests passed.
- No maintained Python script references the former parent workspace.

## Training entry points

Each trainer completed a real one-epoch run on the packaged joint wind data
and wrote a loadable checkpoint to a temporary directory:

| Entry point | Smoke protocol | Status |
|---|---:|---|
| `train_joint.py` | 1 forecast epoch + 1 diffusion epoch | passed |
| `train_st_jcdm.py` | 1 forecast epoch + 1 diffusion epoch | passed |
| `train_joint_wgan_gp.py` | 1 epoch | passed |
| `train_joint_vae.py` | 1 epoch | passed |
| `train_joint_umnn.py` | 1 epoch | passed |
| `train_joint_ddpm.py` | 1 epoch | passed |

These runs validate the complete data, forward, backward, optimizer,
validation, checkpoint-selection, and serialization paths. They do not replace
the full paper epoch counts.

## Evaluation and statistics

The following paths were first executed against the packaged seed-0
checkpoints:

- six-model locked-test evaluation;
- dual-expert locked-test evaluation;
- validation-only dual-expert weight selection;
- Joint CLDM marginal-calibration audit, including five packaged sitewise
  CLDM checkpoints;
- paired significance summary using the frozen ten-seed result.

The frozen reported records remain under `results/metrics/`. After packaging the
remaining final checkpoints, all six model families were additionally loaded
and sampled for seeds 0 through 9 directly from `artifacts/checkpoints/`.
The dual-expert ten-seed path was checked separately. No missing file,
state-dictionary mismatch, or seed mismatch was found.

## Figure generation

The following scripts completed and regenerated their outputs:

- `make_paper_visualizations.py`;
- `make_locked_test_dm_tests.py`;
- `make_fica_backtest_figure.py`.

## FICA and Gurobi

- Gurobi 13.0.2 and the academic license were detected.
- Model scenario-cache preparation passed.
- The formal 50-day orchestrator passed manifest and dry-run validation.
- A reduced five-site FICA problem was solved to Gurobi optimal status through
  `stde_cdm.fica_system`, validating the packaged optimizer, load curve,
  network construction, scenario adapter, and result serialization.
- The native-center witness workflow was also executed with a reduced scenario
  budget. Both its initial-policy solve and constraint-witness re-solve reached
  optimal status, validating candidate screening, witness selection,
  out-of-sample evaluation, real-trajectory evaluation, and restart-safe case
  serialization.
- The notebook now calls the same maintained module. Its packaged IEEE RTS
  24-hour case with 38 generators, five sites and 200 optimization scenarios
  also completed at optimal status; the recorded solve time was 238.98 s and
  the independent-pool reliability was 0.861.

The complete paper-sized 50-day FICA experiment was not rerun during this
audit because it comprises long optimization jobs. Its checkpoint/restart
logic and compact frozen results are retained. Scripts named
`run_fica_constraint_witness_pilot.py`, `run_two_model_witness_day.py`, and
`validate_fica_model_oos.py` are dependent workflow stages: they require the
upstream policy or scenario files produced by the preceding FICA command and
are not intended as first-step standalone commands.
