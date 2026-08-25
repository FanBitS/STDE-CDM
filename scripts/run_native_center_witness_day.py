#!/usr/bin/env python3
"""One-day pilot: each model uses its own native deterministic center."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_fica_constraint_witness_pilot import constraint_witnesses
from run_fica_real_backtest import DATA_FILE, atomic_json, atomic_npz, base_fica_system, load_joint, load_models
from run_two_model_witness_day import MODELS, generate_paired, slug, solve_case, utc_now


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=int, default=18)
    parser.add_argument("--candidate-count", type=int, default=6000)
    parser.add_argument("--validation-count", type=int, default=5000)
    parser.add_argument("--selected-count", type=int, default=200)
    parser.add_argument("--max-witnesses", type=int, default=100)
    parser.add_argument(
        "--stde-center",
        choices=("fused", "spatiotemporal"),
        default="fused",
    )
    parser.add_argument("--only-model", choices=MODELS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--feasibility-tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    day = args.day
    weight = 0.4
    suffix = (
        "two_models"
        if args.stde_center == "fused"
        else "stde_spatiotemporal_center_matched"
    )
    output = (
        args.output_dir
        if args.output_dir is not None
        else ROOT / "outputs" / f"fica_native_center_day_{day:02d}_{suffix}"
    )
    output.mkdir(parents=True, exist_ok=True)
    data = load_joint(DATA_FILE)
    condition_np = data.x_test[day : day + 1]
    observation_pu = np.asarray(data.y_test[day], dtype=np.float32)
    date = str(data.test_dates[day])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)
    base, st = load_models(device)
    condition = torch.from_numpy(condition_np).to(device)
    with torch.no_grad():
        distribution_center = base.forecast(condition)[0].cpu().numpy()
        spatiotemporal_center = st.forecast(condition)[0].cpu().numpy()
    stde_center = (
        (1.0 - weight) * distribution_center
        + weight * spatiotemporal_center
        if args.stde_center == "fused"
        else spatiotemporal_center
    )
    centers = {
        "Joint-CLDM": distribution_center.astype(np.float32),
        "STDE-CDM": stde_center.astype(np.float32),
    }

    initial_seed = 20260725 + day
    candidate_seed = 40260725 + day
    validation_seed = 30260725 + day
    core_seed = 50260725 + day
    pool_file = output / "native_pools.npz"
    matched_pool = (
        ROOT / "outputs" / f"fica_native_center_day_{day:02d}_two_models"
        / "native_pools.npz"
    )
    pool_source = (
        matched_pool
        if args.output_dir is None
        and args.stde_center == "spatiotemporal"
        and matched_pool.is_file()
        else pool_file
    )
    if pool_source.is_file():
        with np.load(pool_source) as pools:
            initial = {m: np.asarray(pools[f"{slug(m)}_initial"]) for m in MODELS}
            candidates = {m: np.asarray(pools[f"{slug(m)}_candidate"]) for m in MODELS}
            validations = {m: np.asarray(pools[f"{slug(m)}_validation"]) for m in MODELS}
        print(f"[pool] loaded exact pools from {pool_source}", flush=True)
    else:
        initial = generate_paired(base, st, condition, args.selected_count, initial_seed, weight)
        candidates = generate_paired(base, st, condition, args.candidate_count, candidate_seed, weight)
        validations = generate_paired(base, st, condition, args.validation_count, validation_seed, weight)
        atomic_npz(
            pool_file,
            day=np.asarray(day),
            date=np.asarray(date),
            initial_seed=np.asarray(initial_seed),
            candidate_seed=np.asarray(candidate_seed),
            validation_seed=np.asarray(validation_seed),
            joint_cldm_center=centers["Joint-CLDM"],
            stde_cdm_center=centers["STDE-CDM"],
            joint_cldm_initial=initial["Joint-CLDM"],
            stde_cdm_initial=initial["STDE-CDM"],
            joint_cldm_candidate=candidates["Joint-CLDM"],
            stde_cdm_candidate=candidates["STDE-CDM"],
            joint_cldm_validation=validations["Joint-CLDM"],
            stde_cdm_validation=validations["STDE-CDM"],
        )
        print("[pool] generated and cached native-center pools", flush=True)
    del base, st
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fica_args = SimpleNamespace(
        scenarios=args.selected_count, epsilon=0.03, theta=0.06,
        wind_share=0.45, time_limit=14400.0,
    )
    template = base_fica_system(fica_args)
    wind_capacity = float(template.pop("wind_capacity_mw"))
    per_farm_capacity = wind_capacity / 5
    results: dict[str, dict] = {}

    selected_models = (args.only_model,) if args.only_model else MODELS
    for model_name in selected_models:
        forecast = centers[model_name]
        model_results: dict[str, dict] = {}
        initial_result = solve_case(
            output, template, model_name, "Initial-200",
            initial[model_name], forecast, observation_pu,
            validations[model_name], per_farm_capacity, date,
            {
                "name": "native-center direct paired random 200",
                "scenario_seed": initial_seed,
                "forecast_source": (
                    "Joint CLDM distribution expert"
                    if model_name == "Joint-CLDM"
                    else (
                        "spatiotemporal expert joint forecast head"
                        if args.stde_center == "spatiotemporal"
                        else "0.6 distribution expert + 0.4 spatiotemporal expert"
                    )
                ),
                "uses_observation": False,
            }, tolerance=args.feasibility_tolerance,
        )
        model_results["Initial-200"] = initial_result
        with np.load(output / "cases" / f"{slug(model_name)}_initial_200.npz") as policy:
            generation = np.asarray(policy["scheduled_generation_mw"], dtype=float)
            alpha = np.asarray(policy["alpha"], dtype=float)

        candidate_mw = np.asarray(candidates[model_name], dtype=float) * per_farm_capacity
        witnesses, diagnostics, maximum_violation = constraint_witnesses(
            template, generation, alpha, forecast * per_farm_capacity, candidate_mw,
            tolerance=args.feasibility_tolerance,
        )
        if len(witnesses) > args.max_witnesses:
            ranked = sorted(witnesses, key=lambda i: maximum_violation[i], reverse=True)
            witnesses = np.asarray(ranked[: args.max_witnesses], dtype=int)
        excluded = np.ones(args.candidate_count, dtype=bool)
        excluded[witnesses] = False
        available = np.flatnonzero(excluded)
        core_count = args.selected_count - len(witnesses)
        rng = np.random.RandomState(core_seed)
        core = rng.choice(available, size=core_count, replace=False)
        selected_indices = np.concatenate([witnesses, core])
        selected = candidates[model_name][selected_indices]
        selection = {
            "name": "native-center constraint witnesses plus uniform core",
            "candidate_count": args.candidate_count,
            "candidate_seed": candidate_seed,
            "witness_count": int(len(witnesses)),
            "core_count": int(core_count),
            "core_seed": core_seed,
            "maximum_witness_budget": args.max_witnesses,
            "forecast_source": (
                "Joint CLDM distribution expert"
                if model_name == "Joint-CLDM"
                else (
                    "spatiotemporal expert joint forecast head"
                    if args.stde_center == "spatiotemporal"
                    else "0.6 distribution expert + 0.4 spatiotemporal expert"
                )
            ),
            "uses_observation": False,
            "witness_diagnostics": diagnostics,
            "candidate_scenarios_violating_initial_policy": int(
                np.count_nonzero(
                    maximum_violation > args.feasibility_tolerance
                )
            ),
        }
        witness_result = solve_case(
            output, template, model_name, "Witness-200",
            selected, forecast, observation_pu, validations[model_name],
            per_farm_capacity, date, selection,
            tolerance=args.feasibility_tolerance,
        )
        model_results["Witness-200"] = witness_result
        results[model_name] = model_results

    summary = {
        "complete": True,
        "completed_utc": utc_now(),
        "experiment": "native deterministic center per complete model",
        "day_index": day,
        "date": date,
        "weight_spatiotemporal": weight,
        "stde_center": args.stde_center,
        "feasibility_tolerance_mw": args.feasibility_tolerance,
        "solve_count": 4,
        "results": results,
    }
    atomic_json(output / "summary.json", summary)
    print("\nFINAL NATIVE-CENTER COMPARISON", flush=True)
    for model_name in selected_models:
        for method in ("Initial-200", "Witness-200"):
            row = results[model_name][method]
            print(
                f"{model_name:12s} {method:11s} "
                f"JCC={row['independent_5000_validation']['joint_jcc']:.2%} "
                f"real={row['real_trajectory']['joint_feasible']} "
                f"cost={row['real_trajectory']['realized_cost']:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
