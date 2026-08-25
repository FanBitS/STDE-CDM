# STDE-CDM

Official implementation of **STDE-CDM**, a spatiotemporal dual expert
conditional diffusion model for synchronized multisite wind scenario
generation.

STDE-CDM combines a sitewise distribution expert with a joint spatiotemporal
expert. The two experts learn complementary conditional residual
distributions, use paired Gaussian innovations during sampling, and fuse their
reconstructed trajectories into one synchronized multivariate scenario. The
repository also provides five jointly trained baselines and a downstream FICA
distributionally robust dispatch evaluation.

## Highlights

- Joint generation of complete `24 hours × K sites` trajectories.
- Sitewise distribution modeling and explicit cross-site temporal modeling.
- Paired reverse diffusion with shared innovations and trajectory-level fusion.
- Matched implementations of Joint WGAN GP, Joint VAE, Joint UMNN, Joint DDPM,
  and Joint CLDM.
- Ten-seed evaluation on 50 locked TEST days with 200 scenarios per day.
- A 50-day FICA dispatch backtest on the IEEE RTS 24 system.

## Locked-test results

The table reports mean ± standard deviation over ten matched training seeds.
Every model is evaluated on the same 50 locked TEST days using 200 synchronized
scenarios per day. Lower values are better for all metrics.

| Model | MAE | RMSE | CRPS | PS | ES | VS |
|---|---:|---:|---:|---:|---:|---:|
| Joint WGAN GP | 0.21263 ± 0.01912 | 0.27070 ± 0.02059 | 0.19817 ± 0.02605 | 0.10027 ± 0.01250 | 2.65719 ± 0.28339 | 1018.89 ± 205.61 |
| Joint VAE | 0.15163 ± 0.00165 | 0.19374 ± 0.00177 | 0.11961 ± 0.00118 | 0.06206 ± 0.00062 | 1.64576 ± 0.01465 | 607.71 ± 6.41 |
| Joint UMNN | 0.12875 ± 0.00067 | 0.17601 ± 0.00127 | 0.09208 ± 0.00050 | 0.04822 ± 0.00026 | 1.32350 ± 0.00873 | 493.13 ± 2.94 |
| Joint DDPM | 0.12180 ± 0.00175 | 0.16128 ± 0.00243 | 0.08549 ± 0.00151 | 0.04481 ± 0.00080 | 1.22312 ± 0.02046 | 445.03 ± 6.44 |
| Joint CLDM | 0.11279 ± 0.00157 | 0.15476 ± 0.00144 | 0.08095 ± 0.00111 | 0.04239 ± 0.00058 | 1.18289 ± 0.01364 | 426.55 ± 5.98 |
| **STDE-CDM** | **0.11007 ± 0.00115** | **0.15098 ± 0.00085** | **0.07840 ± 0.00058** | **0.04107 ± 0.00030** | **1.14778 ± 0.00662** | **411.18 ± 2.55** |

The formal FICA protocol uses each model's native deterministic forecast and
selects 200 constraint-relevant trajectories from its 6,000-scenario candidate
distribution. Joint CLDM and STDE-CDM both satisfy the joint real-trajectory
constraints on 45 of 50 days. STDE-CDM lowers the mean realized operating cost
from 821,394.04 to 820,554.09 USD per day and has lower cost on 48 of 50 days.

The authoritative machine-readable records are
`results/metrics/paper_locked_test_10seed.json` and
`results/fica/final_summary.json`.

## Repository layout

```text
STDE-CDM/
├── src/                       # models, metrics, data, and FICA adapter
├── scripts/                   # training, evaluation, plotting, and dispatch
├── configs/                   # frozen experiment configurations
├── tests/                     # automated model and interface checks
├── data/                      # aligned wind data and compact scenario pool
├── artifacts/checkpoints/     # final checkpoints for all ten seeds
├── results/                   # frozen metrics and compact FICA records
├── figures/results/           # reproducible quantitative figures
├── notebooks/                 # inspectable wind-to-FICA trace
├── fica_dispatch_optimizer/   # downstream optimizer and upstream notice
├── docs/                      # experiment and verification protocols
├── environment.yml
└── pyproject.toml
```

## Installation

The reference environment uses Python 3.11 and PyTorch 2.1 or newer.

```bash
git clone git@github.com:FanBitS/STDE-CDM.git
cd STDE-CDM
conda env create -f environment.yml
conda activate stde-cdm
```

For an existing Python environment:

```bash
python -m pip install -e ".[plot,test,notebook]"
```

Fresh FICA solves additionally require Gurobi and a valid Gurobi license:

```bash
python -m pip install -e ".[dispatch,plot,test,notebook]"
```

Scenario generation and the six-metric evaluation do not require Gurobi.

## Quick verification

```bash
make compile
make test
```

The latest full entry-point audit is recorded in
[`docs/SCRIPT_VERIFICATION.md`](docs/SCRIPT_VERIFICATION.md).

## Training

All trainers use the same data split and receive one joint target tensor with
shape `24 × K`.

```bash
python scripts/train_joint_wgan_gp.py --seed 0
python scripts/train_joint_vae.py --seed 0
python scripts/train_joint_umnn.py --seed 0
python scripts/train_joint_ddpm.py --seed 0
python scripts/train_joint.py --seed 0
python scripts/train_st_jcdm.py --seed 0
```

Repeat with seeds 0 through 9 for the complete protocol. New checkpoints are
written below `outputs/checkpoints/`. The final ten-seed checkpoints are already
provided in `artifacts/checkpoints/`, so the reported metrics can be reproduced
without retraining.

## Evaluation

```bash
python scripts/evaluate_all_models_locked_test.py \
  --seeds 0 1 2 3 4 5 6 7 8 9 --scenarios 200

python scripts/evaluate_dual_expert_locked_test.py \
  --seeds 0 1 2 3 4 5 6 7 8 9 --scenarios 200 --weight 0.4

python scripts/build_paper_locked_test_result.py
python scripts/summarize_paired_metrics.py
```

Rebuild the quantitative figures:

```bash
make figures
make fica-figure
```

All generated figures are written to `figures/results/`.

## FICA dispatch validation

Inspect the formal 50-day run without solving:

```bash
python scripts/run_fica_native_sthead_50day.py --dry-run
```

Run selected days or the full locked backtest:

```bash
python scripts/run_fica_native_sthead_50day.py --days 0 1
python scripts/run_fica_native_sthead_50day.py
```

Each case is committed independently. Repeating the same command resumes an
interrupted run while preserving completed cases. Large regenerable candidate
pools are excluded; compact formal records are retained under `results/fica/`.

## Data

`data/wind_data_all_zone.csv` contains the aligned multisite wind data used by
the experiments. It originates from the public
[GEFCom2014 wind forecasting dataset](https://ieee-pes-data-sharing.org/datasets/detail/0e87366e-2e91-4024-b658-43f6b22faa69).
The frozen split contains 631 learning days, 50 validation days, and 50 locked
TEST days. Dataset provenance and integrity information are documented in
[`data/README.md`](data/README.md) and [`PROJECT_MANIFEST.md`](PROJECT_MANIFEST.md).

## Reproducibility and provenance

- Random seeds, splits, scenario counts, and dispatch settings are frozen in
  `configs/`.
- Final checkpoints and reported metric records are versioned in the repository.
- File hashes in the manifests detect accidental artifact changes; they are not
  passwords or encryption keys.
- Third-party origins and retained license notices are listed in
  [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Citation

Please cite this repository using [`CITATION.cff`](CITATION.cff). The preferred
article citation will be added after publication metadata is available.

## License status

A project-wide license for the newly authored STDE-CDM code has not yet been
selected. The retained FICA optimizer remains subject to its upstream GPL 3.0
notice, and other inherited components retain their original terms. See
[`THIRD_PARTY.md`](THIRD_PARTY.md) before redistribution.
