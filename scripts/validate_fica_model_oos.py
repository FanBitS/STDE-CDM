#!/usr/bin/env python3
"""Validate saved FICA policies on independent model-generated scenarios.

This script does not solve FICA again.  For one locked TEST day it:

1. loads the two policies previously optimized with 200 scenarios;
2. independently generates a larger same-day scenario set from each model;
3. evaluates generation, line-flow, ramping, balance, cost, and joint JCC;
4. compares the observed trajectory with the independent model distribution.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(ROOT / "fica_dispatch_optimizer"),
]

from run_fica_real_backtest import (  # noqa: E402
    DATA_FILE,
    MODELS,
    base_fica_system,
    load_joint,
    load_models,
)
from solar_all_method import check_JCC  # noqa: E402


DEFAULT_BACKTEST = ROOT / "outputs" / "fica_real_backtest_seed0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent same-day model-distribution JCC validation"
    )
    parser.add_argument("--day", type=int, default=4)
    parser.add_argument("--scenarios", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=30260725)
    parser.add_argument("--weight", type=float, default=0.4)
    parser.add_argument("--backtest", type=Path, default=DEFAULT_BACKTEST)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.stem + ".", suffix=".npz.tmp", delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def wilson_interval(successes: int, count: int, z: float = 1.9599639845) -> list[float]:
    proportion = successes / count
    denominator = 1.0 + z**2 / count
    center = (proportion + z**2 / (2.0 * count)) / denominator
    radius = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / count
            + z**2 / (4.0 * count**2)
        )
        / denominator
    )
    return [float(center - radius), float(center + radius)]


def summarize_violations(maximum: np.ndarray) -> dict[str, float]:
    maximum = np.asarray(maximum, dtype=float)
    return {
        "mean_mw": float(maximum.mean()),
        "p95_mw": float(np.quantile(maximum, 0.95)),
        "p99_mw": float(np.quantile(maximum, 0.99)),
        "maximum_mw": float(maximum.max()),
    }


def evaluate_scenarios(
    system: dict[str, Any],
    scheduled_generation: np.ndarray,
    alpha: np.ndarray,
    forecast_mw: np.ndarray,
    scenarios_mw: np.ndarray,
    tolerance: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    errors = scenarios_mw - forecast_mw[None, :, :]
    total_errors = errors.sum(axis=2)
    actual_generation = (
        scheduled_generation[None, :, :]
        - alpha[None, :, :] * total_errors[:, :, None]
    )

    generation_violation = np.maximum(
        np.maximum(
            actual_generation - system["gen_cap_individual"][None, None, :],
            system["gen_pmin_individual"][None, None, :] - actual_generation,
        ),
        0.0,
    )
    generation_max = generation_violation.max(axis=(1, 2))
    generation_safe = generation_max <= tolerance

    ptdf = np.asarray(system["PTDF"], dtype=float).copy()
    ptdf[np.abs(ptdf) < 1e-5] = 0.0
    ptdf_generation = ptdf[:, system["gen_bus_list"]].T
    ptdf_wind = ptdf[:, system["WT_bus_list"]].T
    ptdf_load = ptdf.T
    line_flow = (
        np.einsum("stg,gl->stl", actual_generation, ptdf_generation)
        + np.einsum("stw,wl->stl", scenarios_mw, ptdf_wind)
        - (system["load_bus_all"] @ ptdf_load)[None, :, :]
    )
    line_violation = np.maximum(
        np.abs(line_flow) - system["P_line_limit"][None, None, :], 0.0
    )
    line_max = line_violation.max(axis=(1, 2))
    line_safe = line_max <= tolerance

    ramp = np.diff(actual_generation, axis=1)
    ramp_violation = np.maximum(
        np.abs(ramp) - system["gen_ramp_rate"][None, None, :], 0.0
    )
    ramp_max = ramp_violation.max(axis=(1, 2))
    ramp_safe = ramp_max <= tolerance

    balance_residual = (
        actual_generation.sum(axis=2)
        + scenarios_mw.sum(axis=2)
        - system["load_bus_all"].sum(axis=1)[None, :]
    )
    balance_max = np.abs(balance_residual).max(axis=1)
    balance_safe = balance_max <= tolerance

    # Match the original FICA ``check_JCC`` definition. Power balance is an
    # identity implied by the affine policy (sum(alpha)=1), so it is retained
    # as a numerical diagnostic but is not counted as an extra JCC event.
    joint_safe = generation_safe & line_safe & ramp_safe
    scenario_cost = (
        system["gen_cost"][None, None, :] * actual_generation
        + system["gen_cost_quadra"][None, None, :] * actual_generation**2
    ).sum(axis=(1, 2))

    count = len(scenarios_mw)
    joint_count = int(joint_safe.sum())
    metrics = {
        "scenario_count": count,
        "joint_jcc_definition": (
            "simultaneous generation, line-flow, and ramping feasibility; "
            "power balance is reported separately as a numerical identity"
        ),
        "joint_satisfied_count": joint_count,
        "joint_jcc": float(joint_safe.mean()),
        "joint_jcc_wilson_95_interval": wilson_interval(joint_count, count),
        "generation_jcc": float(generation_safe.mean()),
        "line_flow_jcc": float(line_safe.mean()),
        "ramping_jcc": float(ramp_safe.mean()),
        "balance_jcc": float(balance_safe.mean()),
        "generation_violation": summarize_violations(generation_max),
        "line_flow_violation": summarize_violations(line_max),
        "ramping_violation": summarize_violations(ramp_max),
        "balance_violation": summarize_violations(balance_max),
        "realized_cost": {
            "mean": float(scenario_cost.mean()),
            "standard_deviation": float(scenario_cost.std(ddof=1)),
            "q05": float(np.quantile(scenario_cost, 0.05)),
            "median": float(np.median(scenario_cost)),
            "q95": float(np.quantile(scenario_cost, 0.95)),
        },
    }
    arrays = {
        "scenario_errors_mw": errors,
        "scenario_total_errors_mw": total_errors,
        "joint_safe": joint_safe,
        "generation_safe": generation_safe,
        "line_flow_safe": line_safe,
        "ramping_safe": ramp_safe,
        "balance_safe": balance_safe,
        "maximum_generation_violation_mw": generation_max,
        "maximum_line_flow_violation_mw": line_max,
        "maximum_ramping_violation_mw": ramp_max,
        "maximum_balance_violation_mw": balance_max,
        "scenario_realized_cost": scenario_cost,
    }
    return metrics, arrays


def observed_coverage(
    scenario_total_errors: np.ndarray, observed_total_error: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    percentiles = np.mean(
        scenario_total_errors <= observed_total_error[None, :], axis=0
    )
    minimum = scenario_total_errors.min(axis=0)
    maximum = scenario_total_errors.max(axis=0)
    q015 = np.quantile(scenario_total_errors, 0.015, axis=0)
    q985 = np.quantile(scenario_total_errors, 0.985, axis=0)
    outside_full = (observed_total_error < minimum) | (
        observed_total_error > maximum
    )
    outside_central_97 = (observed_total_error < q015) | (
        observed_total_error > q985
    )
    metrics = {
        "hours_outside_full_5000_scenario_range": int(outside_full.sum()),
        "hours_outside_central_97_percent_interval": int(
            outside_central_97.sum()
        ),
        "minimum_empirical_percentile": float(percentiles.min()),
        "maximum_empirical_percentile": float(percentiles.max()),
        "hourly_empirical_percentiles": percentiles.tolist(),
    }
    return metrics, percentiles


def main() -> None:
    args = parse_args()
    data = load_joint(DATA_FILE)
    if args.day < 0 or args.day >= len(data.test_dates):
        raise ValueError(f"day must lie in [0, {len(data.test_dates) - 1}]")

    cache_path = args.backtest / "scenario_cache" / f"day_{args.day:02d}.npz"
    with np.load(cache_path) as cache:
        forecast_pu = np.asarray(cache["common_forecast_pu"], dtype=float)
        observation_pu = np.asarray(cache["observation_pu"], dtype=float)
        date = str(np.asarray(cache["date"]).item())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)
    base, spatiotemporal = load_models(device)
    condition = torch.from_numpy(data.x_test[args.day : args.day + 1]).to(device)
    evaluation_seed = args.seed + args.day

    torch.manual_seed(evaluation_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(evaluation_seed)
    with torch.no_grad():
        joint_scenarios_pu = (
            base.sample(condition, args.scenarios)[0].cpu().numpy()
        )
        generator = torch.Generator(device=device).manual_seed(evaluation_seed)
        spatiotemporal_scenarios_pu = (
            spatiotemporal.sample(condition, args.scenarios, generator)[0]
            .cpu()
            .numpy()
        )
    stde_scenarios_pu = (
        (1.0 - args.weight) * joint_scenarios_pu
        + args.weight * spatiotemporal_scenarios_pu
    )
    del base, spatiotemporal
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fica_args = SimpleNamespace(
        scenarios=200,
        epsilon=0.03,
        theta=0.06,
        wind_share=0.45,
        time_limit=14400.0,
    )
    system = base_fica_system(fica_args)
    wind_capacity_mw = float(system.pop("wind_capacity_mw"))
    per_farm_capacity_mw = wind_capacity_mw / forecast_pu.shape[1]
    forecast_mw = forecast_pu * per_farm_capacity_mw
    observation_mw = observation_pu * per_farm_capacity_mw
    observed_total_error = (observation_mw - forecast_mw).sum(axis=1)

    generated = {
        "Joint-CLDM": joint_scenarios_pu,
        "STDE-CDM": stde_scenarios_pu,
    }
    output_dir = (
        args.backtest
        / "model_oos_validation"
        / f"day_{args.day:02d}_n{args.scenarios}"
    )
    summary: dict[str, Any] = {
        "complete": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "day_index": args.day,
        "date": date,
        "evaluation_seed": evaluation_seed,
        "independent_scenario_count": args.scenarios,
        "policy_training_scenario_count": 200,
        "weight_spatiotemporal": args.weight,
        "forecast_source": "seed-0 Joint CLDM distribution expert",
        "models": {},
    }

    stored_arrays: dict[str, Any] = {
        "day_index": np.asarray(args.day),
        "date": np.asarray(date),
        "evaluation_seed": np.asarray(evaluation_seed),
        "common_forecast_pu": forecast_pu.astype(np.float32),
        "observation_pu": observation_pu.astype(np.float32),
        "observed_total_error_mw": observed_total_error,
    }
    for model_name in MODELS:
        slug = model_name.lower().replace("-", "_")
        policy_path = args.backtest / "cases" / f"day_{args.day:02d}_{slug}.npz"
        policy_json_path = (
            args.backtest / "cases" / f"day_{args.day:02d}_{slug}.json"
        )
        with np.load(policy_path) as policy:
            scheduled_generation = np.asarray(
                policy["scheduled_generation_mw"], dtype=float
            )
            alpha = np.asarray(policy["alpha"], dtype=float)
            training_errors_mw = np.asarray(
                policy["training_errors_mw"], dtype=float
            )
        policy_summary = json.loads(policy_json_path.read_text(encoding="utf-8"))

        # Promote cached float32 samples before physical-unit arithmetic so
        # the power-balance identity is not obscured by rounding at ~1e-5 MW.
        scenarios_pu = np.asarray(generated[model_name], dtype=np.float64)
        scenarios_mw = scenarios_pu * per_farm_capacity_mw
        metrics, arrays = evaluate_scenarios(
            system,
            scheduled_generation,
            alpha,
            forecast_mw,
            scenarios_mw,
            args.tolerance,
        )
        coverage, observed_percentiles = observed_coverage(
            arrays["scenario_total_errors_mw"], observed_total_error
        )
        canonical_jcc = check_JCC(
            24,
            38,
            len(system["P_line_limit"]),
            scheduled_generation,
            alpha,
            system["load_bus_all"],
            np.asarray(system["PTDF"], dtype=float).copy(),
            system["gen_cap_individual"],
            system["gen_pmin_individual"],
            forecast_mw,
            arrays["scenario_errors_mw"],
            system["P_line_limit"],
            system["gen_bus_list"],
            system["WT_bus_list"],
            gen_ramp_rate=system["gen_ramp_rate"],
        )
        training_jcc = check_JCC(
            24,
            38,
            len(system["P_line_limit"]),
            scheduled_generation,
            alpha,
            system["load_bus_all"],
            np.asarray(system["PTDF"], dtype=float).copy(),
            system["gen_cap_individual"],
            system["gen_pmin_individual"],
            forecast_mw,
            training_errors_mw,
            system["P_line_limit"],
            system["gen_bus_list"],
            system["WT_bus_list"],
            gen_ramp_rate=system["gen_ramp_rate"],
        )
        metrics["canonical_check_JCC"] = float(canonical_jcc)
        metrics["policy_training_200_JCC"] = float(training_jcc)
        metrics["observed_trajectory_coverage"] = coverage
        metrics["observed_real_backtest_joint_feasible"] = bool(
            policy_summary["real_backtest"]["joint_feasible"]
        )
        metrics["observed_real_backtest"] = policy_summary["real_backtest"]
        summary["models"][model_name] = metrics

        stored_arrays[f"{slug}_scenarios_pu"] = scenarios_pu.astype(np.float32)
        stored_arrays[f"{slug}_scheduled_generation_mw"] = scheduled_generation
        stored_arrays[f"{slug}_alpha"] = alpha
        stored_arrays[f"{slug}_observed_hourly_percentile"] = observed_percentiles
        for key, value in arrays.items():
            stored_arrays[f"{slug}_{key}"] = value

        print(
            f"[{model_name}] JCC={metrics['joint_jcc']:.4%} "
            f"P={metrics['generation_jcc']:.4%} "
            f"L={metrics['line_flow_jcc']:.4%} "
            f"R={metrics['ramping_jcc']:.4%} "
            f"real_safe={metrics['observed_real_backtest_joint_feasible']} "
            f"real_outside_5000_hours="
            f"{coverage['hours_outside_full_5000_scenario_range']}",
            flush=True,
        )

    summary["complete"] = True
    summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_npz(output_dir / "validation_arrays.npz", **stored_arrays)
    atomic_json(output_dir / "summary.json", summary)
    print(f"[saved] {output_dir}", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
