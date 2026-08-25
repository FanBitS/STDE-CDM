#!/usr/bin/env python3
"""One-solve day-04 pilot using constraint-witness scenario selection.

The saved random-200 FICA policy screens an independent 6000-scenario STDE-CDM
candidate pool.  For every generation, ramping, and line-flow constraint that
can be violated, the candidate producing its largest violation is retained.
The remaining slots are filled by uniform candidates.  No observation is used
until the final real-trajectory backtest.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import gurobipy as gp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_fica_real_backtest import (  # noqa: E402
    atomic_json,
    atomic_npz,
    base_fica_system,
    evaluate_real_trajectory,
)
from solar_all_method import check_JCC, solve_PD  # noqa: E402
from validate_fica_model_oos import evaluate_scenarios  # noqa: E402


DAY = 4
SELECTED_COUNT = 200
CORE_SEED = 50260725 + DAY
BACKTEST = ROOT / "outputs" / "fica_real_backtest_seed0"
BALANCED_PILOT = (
    BACKTEST / "balanced_selection_pilot" / "day_04_stde_n6000_m200"
)
OUTPUT = BACKTEST / "constraint_witness_pilot" / "day_04_stde_n6000_m200"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def constraint_witnesses(
    system: dict,
    scheduled_generation: np.ndarray,
    alpha: np.ndarray,
    forecast_mw: np.ndarray,
    candidates_mw: np.ndarray,
    tolerance: float = 1e-5,
) -> tuple[np.ndarray, dict[str, dict[str, int]], np.ndarray]:
    total_error = (candidates_mw - forecast_mw[None, :, :]).sum(axis=2)
    actual_generation = (
        scheduled_generation[None, :, :]
        - alpha[None, :, :] * total_error[:, :, None]
    )
    generation_upper = np.maximum(
        actual_generation - system["gen_cap_individual"][None, None, :], 0.0
    )
    generation_lower = np.maximum(
        system["gen_pmin_individual"][None, None, :] - actual_generation, 0.0
    )
    ramp = np.diff(actual_generation, axis=1)
    ramp_upper = np.maximum(
        ramp - system["gen_ramp_rate"][None, None, :], 0.0
    )
    ramp_lower = np.maximum(
        -ramp - system["gen_ramp_rate"][None, None, :], 0.0
    )

    ptdf = np.asarray(system["PTDF"], dtype=float).copy()
    ptdf[np.abs(ptdf) < 1e-5] = 0.0
    ptdf_generation = ptdf[:, system["gen_bus_list"]].T
    ptdf_wind = ptdf[:, system["WT_bus_list"]].T
    ptdf_load = ptdf.T
    line_flow = (
        np.einsum("stg,gl->stl", actual_generation, ptdf_generation)
        + np.einsum("stw,wl->stl", candidates_mw, ptdf_wind)
        - (system["load_bus_all"] @ ptdf_load)[None, :, :]
    )
    line_upper = np.maximum(
        line_flow - system["P_line_limit"][None, None, :], 0.0
    )
    line_lower = np.maximum(
        -line_flow - system["P_line_limit"][None, None, :], 0.0
    )

    violation_arrays = {
        "generation_upper": generation_upper,
        "generation_lower": generation_lower,
        "ramp_upper": ramp_upper,
        "ramp_lower": ramp_lower,
        "line_upper": line_upper,
        "line_lower": line_lower,
    }
    witnesses: set[int] = set()
    diagnostics: dict[str, dict[str, int]] = {}
    scenario_maxima = []
    for name, values in violation_arrays.items():
        flattened = values.reshape(len(values), -1)
        active = flattened.max(axis=0) > tolerance
        if np.any(active):
            category_witnesses = np.unique(
                np.argmax(flattened[:, active], axis=0)
            )
        else:
            category_witnesses = np.empty(0, dtype=int)
        witnesses.update(int(index) for index in category_witnesses)
        per_scenario = flattened.max(axis=1)
        scenario_maxima.append(per_scenario)
        diagnostics[name] = {
            "violating_candidates": int(np.sum(per_scenario > tolerance)),
            "active_constraints": int(active.sum()),
            "unique_witnesses": int(len(category_witnesses)),
        }
    maximum_violation = np.maximum.reduce(scenario_maxima)
    return np.asarray(sorted(witnesses), dtype=int), diagnostics, maximum_violation


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result_json = OUTPUT / "result.json"
    result_npz = OUTPUT / "result_arrays.npz"
    if result_json.is_file() and result_npz.is_file():
        payload = json.loads(result_json.read_text(encoding="utf-8"))
        if payload.get("complete") is True:
            print("[skip] pilot already complete", flush=True)
            print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
            return

    fica_args = SimpleNamespace(
        scenarios=SELECTED_COUNT,
        epsilon=0.03,
        theta=0.06,
        wind_share=0.45,
        time_limit=14400.0,
    )
    system = base_fica_system(fica_args)
    wind_capacity_mw = float(system.pop("wind_capacity_mw"))
    per_farm_capacity_mw = wind_capacity_mw / 5

    with np.load(BALANCED_PILOT / "result_arrays.npz") as source:
        candidates_pu = np.asarray(source["candidate_scenarios_pu"], dtype=float)
        forecast_pu = np.asarray(source["common_forecast_pu"], dtype=float)
        observation_pu = np.asarray(source["observation_pu"], dtype=float)
        date = str(np.asarray(source["date"]).item())
    candidates_mw = candidates_pu * per_farm_capacity_mw
    forecast_mw = forecast_pu * per_farm_capacity_mw
    observation_mw = observation_pu * per_farm_capacity_mw

    baseline_npz = BACKTEST / "cases" / "day_04_stde_cdm.npz"
    baseline_json = json.loads(
        (BACKTEST / "cases" / "day_04_stde_cdm.json").read_text(encoding="utf-8")
    )
    with np.load(baseline_npz) as baseline:
        baseline_generation = np.asarray(
            baseline["scheduled_generation_mw"], dtype=float
        )
        baseline_alpha = np.asarray(baseline["alpha"], dtype=float)

    witness_indices, witness_diagnostics, baseline_maximum_violation = (
        constraint_witnesses(
            system,
            baseline_generation,
            baseline_alpha,
            forecast_mw,
            candidates_mw,
        )
    )
    if len(witness_indices) > SELECTED_COUNT:
        raise RuntimeError(
            f"{len(witness_indices)} witnesses exceed {SELECTED_COUNT} slots"
        )
    remaining = np.setdiff1d(
        np.arange(len(candidates_pu)), witness_indices, assume_unique=True
    )
    rng = np.random.default_rng(CORE_SEED)
    core_indices = rng.choice(
        remaining, SELECTED_COUNT - len(witness_indices), replace=False
    )
    selected_indices = np.concatenate([witness_indices, core_indices])
    selected_pu = candidates_pu[selected_indices]
    selected_errors_mw = (
        selected_pu - forecast_pu[None, :, :]
    ) * per_farm_capacity_mw
    print(
        f"[selection] witnesses={len(witness_indices)} "
        f"uniform_core={len(core_indices)} total={len(selected_indices)}",
        flush=True,
    )

    system["WT_pred"] = forecast_mw
    system["WT_error_scenarios_train"] = selected_errors_mw
    system["N_WDR"] = SELECTED_COUNT
    system["epsilon"] = 0.03
    system["theta"] = 0.06
    system["time_limit"] = 14400.0
    system["thread"] = 4
    system["rng"] = np.random.RandomState(0)
    system["log_file_name"] = str(OUTPUT / "gurobi.log")
    atomic_json(
        OUTPUT / "running.json",
        {
            "state": "running",
            "day_index": DAY,
            "date": date,
            "model": "STDE-CDM",
            "selection": "constraint witnesses plus uniform core",
            "uses_observation": False,
            "started_utc": utc_now(),
            "pid": os.getpid(),
        },
    )

    started = time.perf_counter()
    result = solve_PD(**system)
    model = result["prob"]
    if model.Status not in {gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT, gp.GRB.SUBOPTIMAL}:
        raise RuntimeError(f"Gurobi status {model.Status}")
    if model.SolCount == 0:
        raise RuntimeError("Gurobi produced no feasible solution")
    scheduled_generation = np.asarray(result["gen_power_all"], dtype=float)
    alpha = np.asarray(result["gen_alpha_all"], dtype=float)
    wall_seconds = time.perf_counter() - started

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
        selected_errors_mw,
        system["P_line_limit"],
        system["gen_bus_list"],
        system["WT_bus_list"],
        gen_ramp_rate=system["gen_ramp_rate"],
    )

    validation_path = (
        BACKTEST
        / "model_oos_validation"
        / "day_04_n5000"
        / "validation_arrays.npz"
    )
    with np.load(validation_path) as validation:
        validation_pu = np.asarray(
            validation["stde_cdm_scenarios_pu"], dtype=float
        )
    validation_metrics, validation_arrays = evaluate_scenarios(
        system,
        scheduled_generation,
        alpha,
        forecast_mw,
        validation_pu * per_farm_capacity_mw,
        1e-5,
    )
    real_metrics, real_arrays = evaluate_real_trajectory(
        system,
        scheduled_generation,
        alpha,
        forecast_mw,
        observation_mw,
    )

    random_validation = json.loads(
        (
            BACKTEST
            / "model_oos_validation"
            / "day_04_n5000"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )["models"]["STDE-CDM"]
    balanced_payload = json.loads(
        (BALANCED_PILOT / "result.json").read_text(encoding="utf-8")
    )

    payload = {
        "complete": True,
        "completed_utc": utc_now(),
        "day_index": DAY,
        "date": date,
        "model": "STDE-CDM",
        "selection": {
            "name": "constraint witnesses plus uniform core",
            "candidate_count": len(candidates_pu),
            "selected_count": SELECTED_COUNT,
            "witness_count": len(witness_indices),
            "uniform_core_count": len(core_indices),
            "core_seed": CORE_SEED,
            "uses_observation": False,
            "witness_diagnostics": witness_diagnostics,
            "candidate_scenarios_violating_baseline": int(
                np.sum(baseline_maximum_violation > 1e-5)
            ),
        },
        "fica": {
            "epsilon": 0.03,
            "theta": 0.06,
            "objective": float(model.ObjVal),
            "status": int(model.Status),
            "gurobi_runtime_seconds": float(model.Runtime),
            "wall_seconds": wall_seconds,
            "training_200_JCC": float(training_jcc),
        },
        "independent_5000_validation": validation_metrics,
        "real_trajectory": real_metrics,
        "baselines": {
            "random_200": {
                "independent_5000_validation": random_validation,
                "real_trajectory": baseline_json["real_backtest"],
            },
            "balanced_200": {
                "independent_5000_validation": balanced_payload[
                    "independent_5000_validation"
                ],
                "real_trajectory": balanced_payload["real_trajectory"],
            },
        },
    }
    atomic_npz(
        result_npz,
        day_index=np.asarray(DAY),
        date=np.asarray(date),
        witness_indices=witness_indices,
        uniform_core_indices=core_indices,
        selected_indices=selected_indices,
        selected_scenarios_pu=selected_pu.astype(np.float32),
        baseline_candidate_maximum_violation_mw=baseline_maximum_violation,
        scheduled_generation_mw=scheduled_generation,
        alpha=alpha,
        **{f"validation_{key}": value for key, value in validation_arrays.items()},
        **{f"real_{key}": value for key, value in real_arrays.items()},
    )
    atomic_json(result_json, payload)
    (OUTPUT / "running.json").unlink(missing_ok=True)
    print(
        f"[result] train200={training_jcc:.2%} "
        f"oos5000={validation_metrics['joint_jcc']:.2%} "
        f"real_safe={real_metrics['joint_feasible']} "
        f"realized_cost={real_metrics['realized_cost']:.3f}",
        flush=True,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
