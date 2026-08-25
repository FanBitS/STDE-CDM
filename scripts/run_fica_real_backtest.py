#!/usr/bin/env python3
"""Restart-safe 50-day real-trajectory FICA backtest.

For every locked TEST day, the script:

1. obtains one common deterministic forecast from the seed-0 distribution
   expert;
2. generates 200 paired Joint CLDM and STDE-CDM scenarios;
3. solves one FICA policy for each model with the same physical system and
   optimization settings;
4. substitutes the observed TEST trajectory into the locked affine AGC policy;
5. records joint and constraint-specific feasibility, violation magnitudes,
   scheduled/realized costs, margins, flows, and adjusted generation.

Every model/day case is committed independently. The NPZ is written first and
the JSON with ``complete: true`` is the commit marker. On restart, completed
cases are skipped; at most the solve active during a power loss is repeated.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gurobipy as gp
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
FICA_OPTIMIZER_DIR = ROOT / "fica_dispatch_optimizer"
sys.path[:0] = [
    str(ROOT / "src"),
    str(FICA_OPTIMIZER_DIR),
]

from stde_cdm import JointCLDM, STJCDM, load_joint  # noqa: E402
from stde_cdm.fica_system import build_parser as fica_parser  # noqa: E402
from stde_cdm.fica_system import build_system  # noqa: E402
from solar_all_method import solve_PD  # noqa: E402


MODELS = ("Joint-CLDM", "STDE-CDM")
DEFAULT_OUTPUT = ROOT / "outputs" / "fica_real_backtest_seed0"
DEFAULT_POOL = (
    ROOT
    / "data"
    / "generated"
    / "wind_UMNN_M_1_z0-1-2-3-4_d0_n6000.npz"
)
DATA_FILE = ROOT / "data" / "wind_data_all_zone.csv"
BASE_CHECKPOINT = (
    ROOT / "artifacts" / "checkpoints" / "joint_cldm_seed0.pt"
)
ST_CHECKPOINT = (
    ROOT / "artifacts" / "checkpoints" / "stde_spatiotemporal_seed0.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restart-safe Joint CLDM vs STDE-CDM real FICA backtest"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--days", type=int, nargs="*", default=None)
    parser.add_argument("--scenarios", type=int, default=200)
    parser.add_argument("--weight", type=float, default=0.4)
    parser.add_argument("--sampling-seed", type=int, default=20260725)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--theta", type=float, default=0.06)
    parser.add_argument("--wind-share", type=float, default=0.45)
    parser.add_argument("--time-limit", type=float, default=14400.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate the cached 200-scenario inputs without solving FICA.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute completed cases (normally they are skipped).",
    )
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temp, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.stem + ".", suffix=".npz.tmp", delete=False
    )
    temp = Path(handle.name)
    handle.close()
    try:
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def is_complete(json_path: Path, npz_path: Path) -> bool:
    if not json_path.is_file() or not npz_path.is_file():
        return False
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("complete") is True


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_models(device: torch.device) -> tuple[JointCLDM, STJCDM]:
    base_payload = torch.load(
        BASE_CHECKPOINT, map_location=device, weights_only=False
    )
    base = JointCLDM().to(device)
    base.load_state_dict(base_payload["state_dict"])
    base.eval()

    st_payload = torch.load(ST_CHECKPOINT, map_location=device, weights_only=False)
    st = STJCDM(**st_payload["config"]).to(device)
    st.load_state_dict(st_payload["state_dict"])
    st.eval()
    return base, st


@torch.no_grad()
def prepare_scenario_cache(
    args: argparse.Namespace,
    day_indices: list[int],
    data: Any,
    device: torch.device,
) -> None:
    cache_dir = args.output / "scenario_cache"
    pending = [
        day
        for day in day_indices
        if args.force or not (cache_dir / f"day_{day:02d}.npz").is_file()
    ]
    if not pending:
        print("[cache] all requested days already exist", flush=True)
        return

    base, st = load_models(device)
    for day in pending:
        started = time.perf_counter()
        condition = torch.from_numpy(data.x_test[day : day + 1]).to(device)
        common_forecast = base.forecast(condition)
        paired_seed = args.sampling_seed + day

        torch.manual_seed(paired_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(paired_seed)
        base_scenarios = base.sample(condition, args.scenarios)
        generator = torch.Generator(device=device).manual_seed(paired_seed)
        st_scenarios = st.sample(condition, args.scenarios, generator)
        fused = (1.0 - args.weight) * base_scenarios + args.weight * st_scenarios

        cache_path = cache_dir / f"day_{day:02d}.npz"
        atomic_npz(
            cache_path,
            day_index=np.asarray(day),
            date=np.asarray(str(data.test_dates[day])),
            paired_sampling_seed=np.asarray(paired_seed),
            nwp=data.x_test[day].astype(np.float32),
            observation_pu=data.y_test[day].astype(np.float32),
            common_forecast_pu=common_forecast[0].cpu().numpy().astype(np.float32),
            joint_cldm_scenarios_pu=base_scenarios[0].cpu().numpy().astype(
                np.float32
            ),
            st_expert_scenarios_pu=st_scenarios[0].cpu().numpy().astype(
                np.float32
            ),
            stde_cdm_scenarios_pu=fused[0].cpu().numpy().astype(np.float32),
        )
        print(
            f"[cache] day={day:02d} seed={paired_seed} "
            f"elapsed={time.perf_counter() - started:.2f}s -> {cache_path}",
            flush=True,
        )

    del base, st
    if device.type == "cuda":
        torch.cuda.empty_cache()


def base_fica_system(args: argparse.Namespace) -> dict[str, Any]:
    parsed = fica_parser().parse_args(
        [
            "--scenario",
            str(DEFAULT_POOL),
            "--train-pool-size",
            "1000",
            "--test-pool-size",
            "5000",
            "--network",
            "case24_ieee_rts",
            "--method",
            "FICA",
            "--T",
            "24",
            "--num-gen",
            "38",
            "--n-wdr",
            str(args.scenarios),
            "--epsilon",
            str(args.epsilon),
            "--theta",
            str(args.theta),
            "--wind-share",
            str(args.wind_share),
            "--seed",
            "0",
            "--time-limit",
            str(args.time_limit),
            "--no-plot",
        ]
    )
    system = build_system(parsed)
    system.pop("WT_error_scenarios_test")
    return system


def array_violation_summary(
    violation: np.ndarray, margin: np.ndarray, tolerance: float
) -> dict[str, Any]:
    violation = np.asarray(violation)
    margin = np.asarray(margin)
    positive = violation[violation > tolerance]
    if violation.ndim == 1:
        violating_hours = int(np.count_nonzero(violation > tolerance))
    else:
        violating_hours = int(
            np.count_nonzero(np.any(violation > tolerance, axis=tuple(range(1, violation.ndim))))
        )
    return {
        "feasible": bool(np.all(violation <= tolerance)),
        "violating_elements": int(np.count_nonzero(violation > tolerance)),
        "violating_hours": violating_hours,
        "max_violation_mw": float(positive.max()) if positive.size else 0.0,
        "mean_positive_violation_mw": (
            float(positive.mean()) if positive.size else 0.0
        ),
        "sum_violation_mw": float(positive.sum()) if positive.size else 0.0,
        "p95_positive_violation_mw": (
            float(np.percentile(positive, 95)) if positive.size else 0.0
        ),
        "minimum_margin_mw": float(margin.min()),
    }


def evaluate_real_trajectory(
    system: dict[str, Any],
    generation: np.ndarray,
    alpha: np.ndarray,
    forecast_mw: np.ndarray,
    observation_mw: np.ndarray,
    tolerance: float = 1e-5,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    error = observation_mw - forecast_mw
    total_error = error.sum(axis=1)
    actual_generation = generation - alpha * total_error[:, None]

    upper_margin = system["gen_cap_individual"][None, :] - actual_generation
    lower_margin = actual_generation - system["gen_pmin_individual"][None, :]
    generation_margin = np.minimum(upper_margin, lower_margin)
    generation_violation = np.maximum(-generation_margin, 0.0)

    ptdf = np.asarray(system["PTDF"]).copy()
    ptdf[np.abs(ptdf) < 1e-5] = 0.0
    ptdf_gen = ptdf[:, system["gen_bus_list"]].T
    ptdf_wind = ptdf[:, system["WT_bus_list"]].T
    ptdf_load = ptdf.T
    line_flow = (
        actual_generation @ ptdf_gen
        + observation_mw @ ptdf_wind
        - system["load_bus_all"] @ ptdf_load
    )
    line_margin = system["P_line_limit"][None, :] - np.abs(line_flow)
    line_violation = np.maximum(-line_margin, 0.0)

    actual_ramp = np.diff(actual_generation, axis=0)
    ramp_margin = system["gen_ramp_rate"][None, :] - np.abs(actual_ramp)
    ramp_violation = np.maximum(-ramp_margin, 0.0)

    balance_residual = (
        actual_generation.sum(axis=1)
        + observation_mw.sum(axis=1)
        - system["load_bus_all"].sum(axis=1)
    )
    balance_violation = np.abs(balance_residual)

    scheduled_cost_hourly = (
        system["gen_cost"][None, :] * generation
        + system["gen_cost_quadra"][None, :] * generation**2
    ).sum(axis=1)
    realized_cost_hourly = (
        system["gen_cost"][None, :] * actual_generation
        + system["gen_cost_quadra"][None, :] * actual_generation**2
    ).sum(axis=1)

    generation_stats = array_violation_summary(
        generation_violation, generation_margin, tolerance
    )
    line_stats = array_violation_summary(line_violation, line_margin, tolerance)
    ramp_stats = array_violation_summary(ramp_violation, ramp_margin, tolerance)
    balance_stats = array_violation_summary(
        balance_violation, tolerance - np.abs(balance_residual), tolerance
    )
    joint = bool(
        generation_stats["feasible"]
        and line_stats["feasible"]
        and ramp_stats["feasible"]
        and balance_stats["feasible"]
    )

    metrics = {
        "joint_feasible": joint,
        "generation": generation_stats,
        "line_flow": line_stats,
        "ramping": ramp_stats,
        "balance": balance_stats,
        "scheduled_cost": float(scheduled_cost_hourly.sum()),
        "realized_cost": float(realized_cost_hourly.sum()),
        "realized_minus_scheduled_cost": float(
            realized_cost_hourly.sum() - scheduled_cost_hourly.sum()
        ),
        "absolute_agc_energy_mwh": float(
            np.abs(actual_generation - generation).sum()
        ),
        "maximum_agc_adjustment_mw": float(
            np.abs(actual_generation - generation).max()
        ),
        "maximum_absolute_total_wind_error_mw": float(
            np.abs(total_error).max()
        ),
        "mean_absolute_total_wind_error_mw": float(
            np.abs(total_error).mean()
        ),
    }
    arrays = {
        "real_error_mw": error,
        "total_real_error_mw": total_error,
        "actual_generation_mw": actual_generation,
        "generation_margin_mw": generation_margin,
        "generation_violation_mw": generation_violation,
        "line_flow_mw": line_flow,
        "line_margin_mw": line_margin,
        "line_violation_mw": line_violation,
        "actual_ramp_mw": actual_ramp,
        "ramp_margin_mw": ramp_margin,
        "ramp_violation_mw": ramp_violation,
        "balance_residual_mw": balance_residual,
        "scheduled_cost_hourly": scheduled_cost_hourly,
        "realized_cost_hourly": realized_cost_hourly,
    }
    return metrics, arrays


def solve_case(
    args: argparse.Namespace,
    template: dict[str, Any],
    day: int,
    model_name: str,
) -> dict[str, Any]:
    case_stem = f"day_{day:02d}_{model_name.lower().replace('-', '_')}"
    case_dir = args.output / "cases"
    json_path = case_dir / f"{case_stem}.json"
    npz_path = case_dir / f"{case_stem}.npz"
    if not args.force and is_complete(json_path, npz_path):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        print(f"[skip] {case_stem} already complete", flush=True)
        return payload

    cache_path = args.output / "scenario_cache" / f"day_{day:02d}.npz"
    with np.load(cache_path) as cache:
        date = str(np.asarray(cache["date"]).item())
        paired_seed = int(np.asarray(cache["paired_sampling_seed"]).item())
        forecast_pu = np.asarray(cache["common_forecast_pu"], dtype=float)
        observation_pu = np.asarray(cache["observation_pu"], dtype=float)
        scenario_key = (
            "joint_cldm_scenarios_pu"
            if model_name == "Joint-CLDM"
            else "stde_cdm_scenarios_pu"
        )
        scenarios_pu = np.asarray(cache[scenario_key], dtype=float)

    system = deepcopy(template)
    wind_capacity_mw = float(system.pop("wind_capacity_mw"))
    per_farm_capacity_mw = wind_capacity_mw / forecast_pu.shape[1]
    forecast_mw = forecast_pu * per_farm_capacity_mw
    observation_mw = observation_pu * per_farm_capacity_mw
    train_errors_mw = (
        scenarios_pu - forecast_pu[None, :, :]
    ) * per_farm_capacity_mw
    system["WT_pred"] = forecast_mw
    system["WT_error_scenarios_train"] = train_errors_mw
    system["N_WDR"] = args.scenarios
    system["epsilon"] = args.epsilon
    system["theta"] = args.theta
    system["time_limit"] = args.time_limit
    system["thread"] = args.threads
    system["rng"] = np.random.RandomState(0)
    system["log_file_name"] = str(case_dir / "logs" / f"{case_stem}.log")
    Path(system["log_file_name"]).parent.mkdir(parents=True, exist_ok=True)

    running_path = case_dir / f"{case_stem}.running.json"
    atomic_json(
        running_path,
        {
            "state": "running",
            "case": case_stem,
            "day_index": day,
            "date": date,
            "model": model_name,
            "started_utc": utc_now(),
            "pid": os.getpid(),
        },
    )
    started = time.perf_counter()
    try:
        result = solve_PD(**system)
        probability_model = result["prob"]
        if probability_model.Status not in {
            gp.GRB.OPTIMAL,
            gp.GRB.TIME_LIMIT,
            gp.GRB.SUBOPTIMAL,
        } or probability_model.SolCount == 0:
            raise RuntimeError(
                "Gurobi ended without a feasible solution: "
                f"status={probability_model.Status}"
            )
        generation_raw = result["gen_power_all"]
        alpha_raw = result["gen_alpha_all"]
        generation = np.asarray(
            generation_raw.X if hasattr(generation_raw, "X") else generation_raw
        )
        alpha = np.asarray(alpha_raw.X if hasattr(alpha_raw, "X") else alpha_raw)
        metrics, realized_arrays = evaluate_real_trajectory(
            system, generation, alpha, forecast_mw, observation_mw
        )
        wall_seconds = time.perf_counter() - started

        atomic_npz(
            npz_path,
            day_index=np.asarray(day),
            date=np.asarray(date),
            model=np.asarray(model_name),
            paired_sampling_seed=np.asarray(paired_seed),
            common_forecast_pu=forecast_pu.astype(np.float32),
            observation_pu=observation_pu.astype(np.float32),
            model_scenarios_pu=scenarios_pu.astype(np.float32),
            common_forecast_mw=forecast_mw,
            observation_mw=observation_mw,
            training_errors_mw=train_errors_mw,
            scheduled_generation_mw=generation,
            alpha=alpha,
            load_bus_all_mw=system["load_bus_all"],
            gen_capacity_mw=system["gen_cap_individual"],
            gen_pmin_mw=system["gen_pmin_individual"],
            gen_ramp_limit_mw=system["gen_ramp_rate"],
            line_limits_mw=system["P_line_limit"],
            gen_cost=system["gen_cost"],
            gen_cost_quadratic=system["gen_cost_quadra"],
            gen_bus_list=system["gen_bus_list"],
            wind_bus_list=system["WT_bus_list"],
            **realized_arrays,
        )
        payload = {
            "complete": True,
            "case": case_stem,
            "day_index": day,
            "date": date,
            "model": model_name,
            "training_seed": 0,
            "paired_sampling_seed": paired_seed,
            "scenario_count": args.scenarios,
            "weight_spatiotemporal": args.weight,
            "common_forecast_source": "seed-0 Joint CLDM distribution expert",
            "epsilon": args.epsilon,
            "theta": args.theta,
            "wind_share": args.wind_share,
            "wind_capacity_mw": wind_capacity_mw,
            "per_farm_capacity_mw": per_farm_capacity_mw,
            "gurobi_status": int(probability_model.Status),
            "gurobi_solution_count": int(probability_model.SolCount),
            "gurobi_runtime_seconds": float(probability_model.Runtime),
            "solve_time_seconds": float(
                result.get("solve_time", probability_model.Runtime)
            ),
            "case_wall_seconds": wall_seconds,
            "fica_objective": float(probability_model.ObjVal),
            "real_backtest": metrics,
            "npz_file": str(npz_path.resolve()),
            "solver_log": str(Path(system["log_file_name"]).resolve()),
            "completed_utc": utc_now(),
        }
        atomic_json(json_path, payload)
        running_path.unlink(missing_ok=True)
        print(
            f"[done] {case_stem} joint={metrics['joint_feasible']} "
            f"cost={metrics['realized_cost']:.3f} wall={wall_seconds:.1f}s",
            flush=True,
        )
        return payload
    except BaseException as exc:
        atomic_json(
            running_path,
            {
                "state": "interrupted_or_failed",
                "case": case_stem,
                "day_index": day,
                "date": date,
                "model": model_name,
                "started_utc": json.loads(
                    running_path.read_text(encoding="utf-8")
                ).get("started_utc"),
                "updated_utc": utc_now(),
                "pid": os.getpid(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


def flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), item, output)
    elif isinstance(value, (list, tuple)):
        output[prefix] = json.dumps(jsonable(value), ensure_ascii=False)
    else:
        output[prefix] = value


def rebuild_summary(output_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_dir / "cases").glob("day_*.json")):
        if path.name.endswith(".running.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("complete") is True:
            rows.append(payload)

    by_model: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        selected = [row for row in rows if row["model"] == model]
        if not selected:
            continue
        backtests = [row["real_backtest"] for row in selected]
        by_model[model] = {
            "completed_days": len(selected),
            "joint_safe_days": int(
                sum(item["joint_feasible"] for item in backtests)
            ),
            "joint_satisfaction_rate": float(
                np.mean([item["joint_feasible"] for item in backtests])
            ),
            "generation_satisfaction_rate": float(
                np.mean([item["generation"]["feasible"] for item in backtests])
            ),
            "line_flow_satisfaction_rate": float(
                np.mean([item["line_flow"]["feasible"] for item in backtests])
            ),
            "ramping_satisfaction_rate": float(
                np.mean([item["ramping"]["feasible"] for item in backtests])
            ),
            "balance_satisfaction_rate": float(
                np.mean([item["balance"]["feasible"] for item in backtests])
            ),
            "mean_scheduled_cost": float(
                np.mean([item["scheduled_cost"] for item in backtests])
            ),
            "mean_realized_cost": float(
                np.mean([item["realized_cost"] for item in backtests])
            ),
            "std_realized_cost": (
                float(np.std([item["realized_cost"] for item in backtests], ddof=1))
                if len(backtests) > 1
                else 0.0
            ),
            "total_generation_violating_elements": int(
                sum(item["generation"]["violating_elements"] for item in backtests)
            ),
            "total_line_violating_elements": int(
                sum(item["line_flow"]["violating_elements"] for item in backtests)
            ),
            "total_ramp_violating_elements": int(
                sum(item["ramping"]["violating_elements"] for item in backtests)
            ),
            "maximum_generation_violation_mw": float(
                max(item["generation"]["max_violation_mw"] for item in backtests)
            ),
            "maximum_line_violation_mw": float(
                max(item["line_flow"]["max_violation_mw"] for item in backtests)
            ),
            "maximum_ramp_violation_mw": float(
                max(item["ramping"]["max_violation_mw"] for item in backtests)
            ),
            "total_solver_seconds": float(
                sum(row["solve_time_seconds"] for row in selected)
            ),
        }

    paired: dict[str, Any] = {}
    indexed = {(row["day_index"], row["model"]): row for row in rows}
    paired_days = sorted(
        day
        for day in {row["day_index"] for row in rows}
        if all((day, model) in indexed for model in MODELS)
    )
    if paired_days:
        joint_pairs = [
            (
                bool(indexed[(day, MODELS[0])]["real_backtest"]["joint_feasible"]),
                bool(indexed[(day, MODELS[1])]["real_backtest"]["joint_feasible"]),
            )
            for day in paired_days
        ]
        paired = {
            "paired_days": len(paired_days),
            "both_safe": sum(a and b for a, b in joint_pairs),
            "joint_cldm_only_safe": sum(a and not b for a, b in joint_pairs),
            "stde_cdm_only_safe": sum(not a and b for a, b in joint_pairs),
            "both_unsafe": sum(not a and not b for a, b in joint_pairs),
            "mean_realized_cost_difference_stde_minus_joint_cldm": float(
                np.mean(
                    [
                        indexed[(day, MODELS[1])]["real_backtest"]["realized_cost"]
                        - indexed[(day, MODELS[0])]["real_backtest"]["realized_cost"]
                        for day in paired_days
                    ]
                )
            ),
        }

    summary = {
        "updated_utc": utc_now(),
        "completed_cases": len(rows),
        "expected_cases": 100,
        "models": by_model,
        "paired": paired,
    }
    atomic_json(output_dir / "summary.json", summary)

    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat: dict[str, Any] = {}
        flatten("", row, flat)
        flat_rows.append(flat)
    if flat_rows:
        fields = sorted({key for row in flat_rows for key in row})
        csv_path = output_dir / "case_metrics.csv"
        temp_path = csv_path.with_name(csv_path.name + ".tmp")
        with temp_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat_rows)
        os.replace(temp_path, csv_path)
    return summary


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_joint(DATA_FILE)
    total_days = len(data.test_dates)
    day_indices = (
        list(range(total_days))
        if args.days is None or len(args.days) == 0
        else sorted(set(args.days))
    )
    if any(day < 0 or day >= total_days for day in day_indices):
        raise ValueError(f"days must lie in [0, {total_days - 1}]")
    if args.scenarios != 200:
        print(
            f"[warning] paper protocol specifies 200 scenarios, got {args.scenarios}",
            flush=True,
        )

    manifest = {
        "protocol": "50-day real-trajectory FICA backtest",
        "created_or_updated_utc": utc_now(),
        "models": MODELS,
        "training_seed": 0,
        "weight_spatiotemporal": args.weight,
        "scenario_count_per_model_day": args.scenarios,
        "common_forecast_source": "seed-0 Joint CLDM distribution expert",
        "paired_sampling_seed_rule": f"{args.sampling_seed} + day_index",
        "test_days_total": total_days,
        "requested_day_indices": day_indices,
        "epsilon": args.epsilon,
        "theta": args.theta,
        "wind_share": args.wind_share,
        "network": "IEEE RTS-24",
        "fica_method": "FICA",
        "time_limit_seconds": args.time_limit,
        "threads": args.threads,
        "base_checkpoint": str(BASE_CHECKPOINT.resolve()),
        "st_checkpoint": str(ST_CHECKPOINT.resolve()),
        "data_file": str(DATA_FILE.resolve()),
        "resume_rule": (
            "A case is complete only when both its NPZ and its JSON commit "
            "marker with complete=true exist."
        ),
    }
    atomic_json(args.output / "manifest.json", manifest)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)
    prepare_scenario_cache(args, day_indices, data, device)
    if args.prepare_only:
        print("[prepare-only] scenario cache complete", flush=True)
        return

    template = base_fica_system(args)
    for day in day_indices:
        for model_name in MODELS:
            solve_case(args, template, day, model_name)
            summary = rebuild_summary(args.output)
            print(
                f"[progress] {summary['completed_cases']}/"
                f"{summary['expected_cases']} cases",
                flush=True,
            )

    summary = rebuild_summary(args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
