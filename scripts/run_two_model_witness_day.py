#!/usr/bin/env python3
"""Four-solve witness-refinement experiment for one locked TEST day."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import gurobipy as gp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_fica_constraint_witness_pilot import constraint_witnesses  # noqa: E402
from run_fica_real_backtest import (  # noqa: E402
    DATA_FILE,
    atomic_json,
    atomic_npz,
    base_fica_system,
    evaluate_real_trajectory,
    load_joint,
    load_models,
)
from solar_all_method import check_JCC, solve_PD  # noqa: E402
from validate_fica_model_oos import evaluate_scenarios  # noqa: E402

MODELS = ("Joint-CLDM", "STDE-CDM")
METHODS = ("Initial-200", "Witness-200")
BACKTEST = ROOT / "outputs" / "fica_real_backtest_seed0"


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=int, default=7)
    parser.add_argument("--candidate-count", type=int, default=6000)
    parser.add_argument("--validation-count", type=int, default=5000)
    parser.add_argument("--selected-count", type=int, default=200)
    parser.add_argument("--max-witnesses", type=int, default=100)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(value: str) -> str:
    return value.lower().replace("-", "_")


@torch.no_grad()
def generate_paired(
    base,
    spatiotemporal,
    condition: torch.Tensor,
    count: int,
    seed: int,
    weight: float = 0.4,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    if condition.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    joint = base.sample(condition, count)[0].cpu().numpy()
    generator = torch.Generator(device=condition.device).manual_seed(seed)
    st = spatiotemporal.sample(condition, count, generator)[0].cpu().numpy()
    return {
        "Joint-CLDM": joint.astype(np.float32),
        "STDE-CDM": ((1.0 - weight) * joint + weight * st).astype(np.float32),
    }


def solve_case(
    output: Path,
    template: dict,
    model_name: str,
    method_name: str,
    scenarios_pu: np.ndarray,
    forecast_pu: np.ndarray,
    observation_pu: np.ndarray,
    validation_pu: np.ndarray,
    per_farm_capacity: float,
    date: str,
    selection: dict,
    tolerance: float = 1e-5,
) -> dict:
    stem = f"{slug(model_name)}_{slug(method_name)}"
    json_path = output / "cases" / f"{stem}.json"
    npz_path = output / "cases" / f"{stem}.npz"
    if json_path.is_file() and npz_path.is_file():
        saved = json.loads(json_path.read_text(encoding="utf-8"))
        if saved.get("complete") is True:
            print(f"[skip] {stem}", flush=True)
            return saved

    system = deepcopy(template)
    forecast_mw = forecast_pu * per_farm_capacity
    observation_mw = observation_pu * per_farm_capacity
    scenarios_pu = np.asarray(scenarios_pu, dtype=float)
    training_errors_mw = (
        scenarios_pu - forecast_pu[None, :, :]
    ) * per_farm_capacity
    system["WT_pred"] = forecast_mw
    system["WT_error_scenarios_train"] = training_errors_mw
    system["N_WDR"] = len(scenarios_pu)
    system["epsilon"] = 0.03
    system["theta"] = 0.06
    system["time_limit"] = 14400.0
    system["thread"] = 4
    system["rng"] = np.random.RandomState(0)
    log_path = output / "logs" / f"{stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    system["log_file_name"] = str(log_path)

    running = output / "cases" / f"{stem}.running.json"
    atomic_json(
        running,
        {
            "state": "running",
            "model": model_name,
            "method": method_name,
            "date": date,
            "started_utc": utc_now(),
            "pid": os.getpid(),
        },
    )
    started = time.perf_counter()
    result = solve_PD(**system)
    problem = result["prob"]
    if problem.Status not in {
        gp.GRB.OPTIMAL,
        gp.GRB.TIME_LIMIT,
        gp.GRB.SUBOPTIMAL,
    } or problem.SolCount == 0:
        raise RuntimeError(
            f"{stem}: no solution, status={problem.Status}, "
            f"count={problem.SolCount}"
        )
    generation = np.asarray(result["gen_power_all"], dtype=float)
    alpha = np.asarray(result["gen_alpha_all"], dtype=float)
    wall = time.perf_counter() - started
    training_jcc = check_JCC(
        24,
        38,
        len(system["P_line_limit"]),
        generation,
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
    validation_metrics, validation_arrays = evaluate_scenarios(
        system,
        generation,
        alpha,
        forecast_mw,
        np.asarray(validation_pu, dtype=float) * per_farm_capacity,
        tolerance,
    )
    real_metrics, real_arrays = evaluate_real_trajectory(
        system, generation, alpha, forecast_mw, observation_mw, tolerance
    )
    payload = {
        "complete": True,
        "completed_utc": utc_now(),
        "model": model_name,
        "method": method_name,
        "date": date,
        "selection": selection,
        "feasibility_tolerance_mw": tolerance,
        "fica": {
            "objective": float(problem.ObjVal),
            "training_JCC": float(training_jcc),
            "gurobi_runtime_seconds": float(problem.Runtime),
            "wall_seconds": wall,
            "status": int(problem.Status),
        },
        "independent_5000_validation": validation_metrics,
        "real_trajectory": real_metrics,
    }
    atomic_npz(
        npz_path,
        model=np.asarray(model_name),
        method=np.asarray(method_name),
        date=np.asarray(date),
        scenarios_pu=scenarios_pu.astype(np.float32),
        forecast_pu=forecast_pu.astype(np.float32),
        observation_pu=observation_pu.astype(np.float32),
        scheduled_generation_mw=generation,
        alpha=alpha,
        **{f"validation_{key}": value for key, value in validation_arrays.items()},
        **{f"real_{key}": value for key, value in real_arrays.items()},
    )
    atomic_json(json_path, payload)
    running.unlink(missing_ok=True)
    print(
        f"[done] {stem} oos={validation_metrics['joint_jcc']:.2%} "
        f"real={real_metrics['joint_feasible']} "
        f"cost={real_metrics['realized_cost']:.2f}",
        flush=True,
    )
    return payload


def main() -> None:
    args = args_parser()
    day = args.day
    output = ROOT / "outputs" / f"fica_witness_day_{day:02d}_two_models"
    output.mkdir(parents=True, exist_ok=True)
    cache_file = output / "paired_pools.npz"
    data = load_joint(DATA_FILE)
    condition = data.x_test[day : day + 1]
    date = str(data.test_dates[day])

    with np.load(BACKTEST / "scenario_cache" / f"day_{day:02d}.npz") as cache:
        forecast_pu = np.asarray(cache["common_forecast_pu"], dtype=float)
        observation_pu = np.asarray(cache["observation_pu"], dtype=float)
        initial = {
            "Joint-CLDM": np.asarray(
                cache["joint_cldm_scenarios_pu"], dtype=np.float32
            ),
            "STDE-CDM": np.asarray(
                cache["stde_cdm_scenarios_pu"], dtype=np.float32
            ),
        }

    candidate_seed = 40260725 + day
    validation_seed = 30260725 + day
    core_seed = 50260725 + day
    if cache_file.is_file():
        with np.load(cache_file) as pools:
            candidates = {
                model: np.asarray(
                    pools[f"{slug(model)}_candidate"], dtype=np.float32
                )
                for model in MODELS
            }
            validations = {
                model: np.asarray(
                    pools[f"{slug(model)}_validation"], dtype=np.float32
                )
                for model in MODELS
            }
        print("[pool] loaded cached paired pools", flush=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[device] {device}", flush=True)
        base, st = load_models(device)
        tensor_condition = torch.from_numpy(condition).to(device)
        candidates = generate_paired(
            base, st, tensor_condition, args.candidate_count, candidate_seed
        )
        validations = generate_paired(
            base, st, tensor_condition, args.validation_count, validation_seed
        )
        atomic_npz(
            cache_file,
            day=np.asarray(day),
            date=np.asarray(date),
            candidate_seed=np.asarray(candidate_seed),
            validation_seed=np.asarray(validation_seed),
            joint_cldm_candidate=candidates["Joint-CLDM"],
            stde_cdm_candidate=candidates["STDE-CDM"],
            joint_cldm_validation=validations["Joint-CLDM"],
            stde_cdm_validation=validations["STDE-CDM"],
        )
        del base, st
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print("[pool] generated and cached paired pools", flush=True)

    fica_args = SimpleNamespace(
        scenarios=args.selected_count,
        epsilon=0.03,
        theta=0.06,
        wind_share=0.45,
        time_limit=14400.0,
    )
    template = base_fica_system(fica_args)
    wind_capacity = float(template.pop("wind_capacity_mw"))
    per_farm_capacity = wind_capacity / forecast_pu.shape[1]

    results: dict[str, dict[str, dict]] = {}
    for model_index, model_name in enumerate(MODELS):
        results[model_name] = {}
        initial_result = solve_case(
            output,
            template,
            model_name,
            "Initial-200",
            initial[model_name],
            forecast_pu,
            observation_pu,
            validations[model_name],
            per_farm_capacity,
            date,
            {
                "name": "direct paired random 200",
                "scenario_seed": 20260725 + day,
                "uses_observation": False,
            },
        )
        results[model_name]["Initial-200"] = initial_result
        initial_npz = output / "cases" / f"{slug(model_name)}_initial_200.npz"
        with np.load(initial_npz) as policy:
            generation = np.asarray(policy["scheduled_generation_mw"], dtype=float)
            alpha = np.asarray(policy["alpha"], dtype=float)

        candidate_mw = (
            np.asarray(candidates[model_name], dtype=float) * per_farm_capacity
        )
        witnesses, diagnostics, maximum_violation = constraint_witnesses(
            template,
            generation,
            alpha,
            forecast_pu * per_farm_capacity,
            candidate_mw,
        )
        if len(witnesses) > args.max_witnesses:
            order = np.argsort(-maximum_violation[witnesses], kind="stable")
            witnesses = witnesses[order[: args.max_witnesses]]
        remaining = np.setdiff1d(
            np.arange(args.candidate_count), witnesses, assume_unique=True
        )
        rng = np.random.default_rng(core_seed)
        core = rng.choice(
            remaining, args.selected_count - len(witnesses), replace=False
        )
        selected_indices = np.concatenate([witnesses, core])
        selected = candidates[model_name][selected_indices]
        refined_result = solve_case(
            output,
            template,
            model_name,
            "Witness-200",
            selected,
            forecast_pu,
            observation_pu,
            validations[model_name],
            per_farm_capacity,
            date,
            {
                "name": "constraint witnesses plus uniform core",
                "candidate_count": args.candidate_count,
                "candidate_seed": candidate_seed,
                "witness_count": int(len(witnesses)),
                "core_count": int(len(core)),
                "core_seed": core_seed,
                "maximum_witness_budget": args.max_witnesses,
                "uses_observation": False,
                "witness_diagnostics": diagnostics,
                "candidate_scenarios_violating_initial_policy": int(
                    np.sum(maximum_violation > 1e-5)
                ),
            },
        )
        results[model_name]["Witness-200"] = refined_result

    summary = {
        "complete": True,
        "completed_utc": utc_now(),
        "day_index": day,
        "date": date,
        "solve_count": 4,
        "candidate_seed": candidate_seed,
        "validation_seed": validation_seed,
        "core_seed": core_seed,
        "results": results,
    }
    atomic_json(output / "summary.json", summary)
    print("\nFINAL COMPARISON", flush=True)
    for model_name in MODELS:
        for method_name in METHODS:
            row = results[model_name][method_name]
            print(
                f"{model_name:10s} {method_name:11s} "
                f"JCC={row['independent_5000_validation']['joint_jcc']:.2%} "
                f"real={row['real_trajectory']['joint_feasible']} "
                f"cost={row['real_trajectory']['realized_cost']:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
