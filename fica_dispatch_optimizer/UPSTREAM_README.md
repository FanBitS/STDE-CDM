# EIFICA-for-PV

> **Envelope Informed Fast Inner Convex Approximation for Distributionally Robust Day-Ahead AGC Scheduling under PV Uncertainty**

Experiment code for the paper. **EIFICA = Envelope Informed Fast Inner Convex Approximation.** The problem is modeled as a Wasserstein distributionally robust joint chance constraint (DRJCC), and inner-convex-approximation solvers are evaluated on the IEEE case24 / case118 systems for solve time, memory usage, reliability, and optimality.

---

## 1. Repository layout

```
eifica-for-pv/
├── solar_all_method.py          # core algorithm (FICA / EIFICA / CVAR solve + JCC check) *to be uploaded
├── paper_eval.py                # main driver: parameter grid, saves .npy + summary table
├── exp_optimality_run.py        # optimality-gap experiment (EIFICA vs exact big-M ExactLHS)
├── live_summary_watch.py        # side watcher: scans finished .npy and appends to CSV
├── solar_error_generalization.py    # PV error modeling / generalization
├── solar_scenario_gen.py        # generates temporally coupled PV forecast-error samples
├── test_gurobi_limit.py         # checks the Gurobi license configuration
│
├── data/                        # load, PV forecast and error samples + PV error figures
├── Solar_c24_speed/  Solar_c24_exact/  Solar_c24_optgap/   # case24 result groups (.npy)
├── Solar_c118_speed/ Solar_c118_exact/                     # case118 result groups (.npy)
├── result_vis/            # result visualization: EIFICA_result_vis.ipynb -> paper figures
└── master_summary_*.csv         # merged summary table
```

### Scripts

| Purpose | File |
|---|---|
| Core algorithm (solve + JCC check) | `solar_all_method.py` *to be uploaded |
| Main driver (parameter grid) | `paper_eval.py` |
| Optimality gap (EIFICA vs ExactLHS) | `exp_optimality_run.py` |
| Live summary CSV | `live_summary_watch.py` |

### Data and figures

Correspondence between the paper figures, the scripts that produce them, and the data they rely on:

| Paper figure | Script | Output |
|---|---|---|
| Dispatch figure: PV output + generator output / AGC / ramp | `solar_all_method.py` (`__main__`) *to be uploaded | `figure/test/` |
| Result figures: solve time / reliability / peak memory / optimality gap (RTS-24 and IEEE-118) | `result_vis/EIFICA_result_vis.ipynb` | `result_vis/figures/` (`time_*`, `reliability_*`, `memory_*`, `optimality_gap*`) |
| PV error figures: marginal distribution / scenario curves / temporal correlation | `solar_scenario_gen.py` (`plot_paper_figures`), `solar_error_generalization.py` | `data/` (`solar_error_*`) |

Data the figures depend on: `master_summary_*.csv`, the `.npy` files under `Solar_c*/`, and the sample files under `data/` (`solar_forecast_prototype.npy`, `solar_temporal_correlated_error_samples.npy`, etc.).

---

## 2. Methods

Select a method via `paper_eval.py --methods`:

| Method | Description |
|---|---|
| **FICA** | Full-sample inner convex approximation: output constraints pruned by 1D order statistics, power flow / ramp use all scenarios. |
| **EIFICA** | Adds inter-temporal ramp constraints on top of FICA and applies scenario screening to them. |
| **CVAR** | Full-sample CVaR reformulation, used as a scalability baseline (`--methods CVAR`). |

### EIFICA two modes

Controlled by the environment variable `EIFICA_MAX_ITER` (default 30):

| Mode | Switch | Description |
|---|---|---|
| iterative (default) | unset | ρ-residual converges to full-sample FICA; objective / JCC match FICA |
| direct | `EIFICA_MAX_ITER=0` | fast approximation solved once using only the endpoint envelope K⁰ |

> The direct mode is the one-shot variant; it is recorded as `one-shot` in the `eifica_mode` column and in directory names.

Other tunable hyperparameters (leave unset to use the default formula, `k=⌊N·ε⌋`):

| Variable | Default | Meaning |
|---|---|---|
| `EIFICA_MAX_ITER` | 30 | max iterations; `0` = direct |
| `EIFICA_M0` | `max(2k, ⌈0.05N⌉)` | initial K⁰ size (per ramp constraint) |
| `EIFICA_M1` | `max(5, ⌈0.01N⌉)` | samples added per iteration |

---

## 3. Environment and dependencies

Python 3, main dependencies:

```
numpy  scipy  pandas  matplotlib  pandapower  gurobipy  joblib  psutil
```

Install:

```bash
pip install numpy scipy pandas matplotlib pandapower gurobipy joblib psutil
```

Solving requires a Gurobi license; point to the license file via an environment variable:

```bash
export GRB_LICENSE_FILE=/path/to/gurobi.lic
```

> Gurobi is free for academic users.

---

## 4. Parameter grid (`paper_eval.py` default)

- (ε, θ): `(0.08,0.12), (0.05,0.10), (0.10,0.15), (0.03,0.06)`
- N_WDR: `50, 80, 100, 150, 200, 250, 300`
- seeds: `0,1,2,3,4` (training scenarios resampled per seed)
- methods: `FICA, EIFICA` (optionally `CVAR`)
- fixed: `num_gen=38, num_Solar=5, num_WT=0, T=24, load_scaling=1, norm_ord=1`
- network: `--network case24_ieee_rts` or `--network case118`

---

## 5. Running experiments

### 5.1 Single point

```bash
GRB_LICENSE_FILE=/path/to/gurobi.lic PYTHONUTF8=1 python paper_eval.py \
    --network case24_ieee_rts --eps-theta 0.03,0.06 --N 100 --seeds 0 --jobs 1
```

Finished `.npy` files are skipped automatically on re-run.

### 5.2 Full grid

- Speed comparison: run `FICA + EIFICA(direct)` in the same directory and pair them directly.
- Exactness check: run `EIFICA(iterative)` separately to verify convergence to full-sample FICA.

Directory layout:

```
Solar_<net>_<kind>/
├── result_*.npy
├── <net><kind>_csv/live_summary.csv
└── logs/
```

```bash
# ===== case24 =====
mkdir -p Solar_c24_speed/logs Solar_c24_speed/24oneshot_csv
tmux new-session -d -s eifica_c24_speed \
  "cd ~/eifica-for-pv && GRB_LICENSE_FILE=$PWD/gurobi.lic EIFICA_MAX_ITER=0 PYTHONUTF8=1 python paper_eval.py --network case24_ieee_rts --methods FICA,EIFICA --out Solar_c24_speed --jobs 3 > Solar_c24_speed/logs/c24_speed.out 2>&1"
tmux new-session -d -s eifica_c24_livecsv \
  "cd ~/eifica-for-pv && PYTHONUTF8=1 python live_summary_watch.py Solar_c24_speed --csv Solar_c24_speed/24oneshot_csv/live_summary.csv --interval 30 > Solar_c24_speed/logs/c24_live_summary.out 2>&1"

mkdir -p Solar_c24_exact/logs Solar_c24_exact/24exact_csv
tmux new-session -d -s eifica_c24_exact \
  "cd ~/eifica-for-pv && GRB_LICENSE_FILE=$PWD/gurobi.lic PYTHONUTF8=1 python paper_eval.py --network case24_ieee_rts --methods EIFICA --out Solar_c24_exact --jobs 3 > Solar_c24_exact/logs/c24_exact.out 2>&1"

# ===== case118 =====
mkdir -p Solar_c118_speed/logs Solar_c118_speed/118oneshot_csv
tmux new-session -d -s eifica_c118_speed \
  "cd ~/eifica-for-pv && GRB_LICENSE_FILE=$PWD/gurobi.lic EIFICA_MAX_ITER=0 PYTHONUTF8=1 python paper_eval.py --network case118 --methods FICA,EIFICA --out Solar_c118_speed --jobs 3 > Solar_c118_speed/logs/c118_speed.out 2>&1"

mkdir -p Solar_c118_exact/logs Solar_c118_exact/118exact_csv
tmux new-session -d -s eifica_c118_exact \
  "cd ~/eifica-for-pv && GRB_LICENSE_FILE=$PWD/gurobi.lic PYTHONUTF8=1 python paper_eval.py --network case118 --methods EIFICA --out Solar_c118_exact --jobs 3 > Solar_c118_exact/logs/c118_exact.out 2>&1"
```

### 5.3 CVaR baseline (optional)

`paper_eval.py` supports `--methods CVAR` for the CVaR → FICA → EIFICA comparison:

```bash
python paper_eval.py --network case24_ieee_rts --methods CVAR --N 50,80 --out Solar_c24_cvar --jobs 2
```

### 5.4 Optimality gap (EIFICA vs exact ExactLHS)

Quantifies the gap between the EIFICA objective and the exact solution (ExactLHS = exact WDR-JCC reformulation with big-M binaries).

- Parameter grid (hard-coded in the script): `SF ∈ {1.0,1.5,2.0}` × `ε ∈ {0.03,0.06}` × `N ∈ {30,60,90,120}` × `10 seeds` × `{EIFICA, ExactLHS}`
- fixed: `T=3, θ=0.06, num_Solar=5, network=case24_ieee_rts`
- parallelism: `N_PARALLEL=8` in the script; time limit set via `OPTGAP_TIME_LIMIT` (default 600)

```bash
mkdir -p Solar_c24_optgap/logs
tmux new-session -d -s eifica_c24_optgap \
  "cd ~/eifica-for-pv && GRB_LICENSE_FILE=$PWD/gurobi.lic PYTHONUTF8=1 python exp_optimality_run.py > Solar_c24_optgap/logs/optgap.out 2>&1"
```

Results are written to `Solar_c24_optgap/optimality_gap_T3_fixed.csv`.

---

## 6. Output and summary

### 6.1 Per instance (`.npy`)

```
{'min_cost (USD)', 'reliability_test (%)', 't_solve (s)', 'peak_rss_mb',
 'method', 'N_WDR', 'epsilon', 'theta', 'seed', 'eifica_mode', 'eifica_iters', ...}
```

### 6.2 Merged summary table

```bash
python paper_eval.py --summarize-only \
    Solar_c24_speed,Solar_c24_exact,Solar_c118_speed,Solar_c118_exact
# produces master_summary_<timestamp>.csv
```

Columns: `network, method, eifica_mode, epsilon, theta, seed, N_WDR, min_cost, reliability_pct, t_solve_s, peak_rss_mb, eifica_iters, source_dir, ...`. The visualization notebook under `result_vis/` reads this table to produce the figures.

---

## 7. Code availability

This repository provides the experiment drivers, data, and results so the experiment design and result structure can be understood and reproduced.

The core algorithm `solar_all_method.py` (marked *to be uploaded) will be added once the related work is formally published, after which the remaining scripts can run the full pipeline directly. For early access, contact the author or open an issue on the repository.
