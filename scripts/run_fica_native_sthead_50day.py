#!/usr/bin/env python3
"""Formal restart-safe 50-day native-head FICA experiment orchestrator.

Each locked TEST day runs four solves:
Joint CLDM initial/witness 200 and STDE CDM initial/witness 200.  Joint CLDM
uses its distribution forecast head; STDE CDM uses its joint spatiotemporal
forecast head.  Full generated trajectories are unchanged by this decomposition.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "fica_native_sthead_50day_seed0_v1"
DAY_SCRIPT = ROOT / "scripts" / "run_native_center_witness_day.py"
BASE_CHECKPOINT = (
    ROOT / "artifacts" / "checkpoints" / "joint_cldm_seed0.pt"
)
ST_CHECKPOINT = (
    ROOT / "artifacts" / "checkpoints" / "stde_spatiotemporal_seed0.pt"
)
DATA_FILE = ROOT / "data" / "wind_data_all_zone.csv"
MODELS = ("Joint-CLDM", "STDE-CDM")
METHODS = ("Initial-200", "Witness-200")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(prefix: str, value: Any, row: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), item, row)
    elif isinstance(value, (list, tuple)):
        row[prefix] = json.dumps(value, separators=(",", ":"))
    else:
        row[prefix] = value


def load_rows(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for day_dir in sorted((output / "days").glob("day_*")):
        summary_file = day_dir / "summary.json"
        if not summary_file.is_file():
            continue
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if not summary.get("complete"):
            continue
        for model in MODELS:
            for method in METHODS:
                if model not in summary["results"]:
                    continue
                case = summary["results"][model][method]
                row: dict[str, Any] = {
                    "day_index": summary["day_index"],
                    "date": summary["date"],
                    "model": model,
                    "method": method,
                }
                flatten("", case, row)
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    preferred = ["day_index", "date", "model", "method", "complete"]
    fields = [key for key in preferred if key in fields] + [
        key for key in fields if key not in preferred
    ]
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def aggregate(output: Path, requested_days: list[int]) -> dict[str, Any]:
    rows = load_rows(output)
    write_csv(output / "all_case_metrics.csv", rows)
    completed_days = sorted({int(row["day_index"]) for row in rows})
    summary: dict[str, Any] = {
        "updated_utc": utc_now(),
        "requested_days": requested_days,
        "completed_days": completed_days,
        "completed_day_count": len(completed_days),
        "completed_case_count": len(rows),
        "expected_case_count": 4 * len(requested_days),
        "methods": {},
    }
    for method in METHODS:
        summary["methods"][method] = {}
        for model in MODELS:
            selected = [
                row for row in rows
                if row["model"] == model and row["method"] == method
            ]
            def vals(key: str) -> list[float]:
                return [float(row[key]) for row in selected if row.get(key) not in ("", None)]
            summary["methods"][method][model] = {
                "case_count": len(selected),
                "real_joint_safe_days": sum(
                    bool(row.get("real_trajectory.joint_feasible")) for row in selected
                ),
                "real_generation_safe_days": sum(
                    bool(row.get("real_trajectory.generation.feasible")) for row in selected
                ),
                "real_ramping_safe_days": sum(
                    bool(row.get("real_trajectory.ramping.feasible")) for row in selected
                ),
                "real_line_safe_days": sum(
                    bool(row.get("real_trajectory.line_flow.feasible")) for row in selected
                ),
                "real_balance_safe_days": sum(
                    bool(row.get("real_trajectory.balance.feasible")) for row in selected
                ),
                "mean_oos_joint_jcc": mean(vals("independent_5000_validation.joint_jcc")),
                "mean_oos_generation_jcc": mean(vals("independent_5000_validation.generation_jcc")),
                "mean_oos_ramping_jcc": mean(vals("independent_5000_validation.ramping_jcc")),
                "mean_oos_line_jcc": mean(vals("independent_5000_validation.line_flow_jcc")),
                "mean_oos_balance_jcc": mean(vals("independent_5000_validation.balance_jcc")),
                "mean_realized_cost": mean(vals("real_trajectory.realized_cost")),
                "mean_scheduled_cost": mean(vals("real_trajectory.scheduled_cost")),
                "mean_generation_max_violation_mw": mean(
                    vals("real_trajectory.generation.max_violation_mw")
                ),
                "mean_ramping_max_violation_mw": mean(
                    vals("real_trajectory.ramping.max_violation_mw")
                ),
                "mean_line_max_violation_mw": mean(
                    vals("real_trajectory.line_flow.max_violation_mw")
                ),
            }

    # Paired cost comparisons are meaningful especially on days where both
    # methods are operationally feasible under the frozen tolerance.
    for method in METHODS:
        lookup = {
            (int(row["day_index"]), row["model"]): row
            for row in rows if row["method"] == method
        }
        paired = []
        both_safe = []
        for day in completed_days:
            a = lookup.get((day, "Joint-CLDM"))
            b = lookup.get((day, "STDE-CDM"))
            if not a or not b:
                continue
            delta = float(b["real_trajectory.realized_cost"]) - float(
                a["real_trajectory.realized_cost"]
            )
            paired.append(delta)
            if (
                bool(a["real_trajectory.joint_feasible"])
                and bool(b["real_trajectory.joint_feasible"])
            ):
                both_safe.append(delta)
        summary["methods"][method]["paired_cost"] = {
            "paired_day_count": len(paired),
            "mean_stde_minus_joint": mean(paired),
            "stde_lower_cost_days": sum(value < 0 for value in paired),
            "both_safe_day_count": len(both_safe),
            "mean_stde_minus_joint_on_both_safe_days": mean(both_safe),
            "stde_lower_cost_both_safe_days": sum(value < 0 for value in both_safe),
        }
    atomic_json(output / "progress_summary.json", summary)
    return summary


def manifest(output: Path, days: list[int], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment_id": "fica_native_sthead_50day_seed0_v1",
        "created_utc": utc_now(),
        "purpose": "Fair complete-method Joint CLDM versus STDE CDM FICA comparison",
        "locked_test_days": days,
        "model_seed": 0,
        "models": {
            "Joint-CLDM": {
                "deterministic_head": "distribution expert forecast",
                "checkpoint": str(BASE_CHECKPOINT.resolve()),
                "sha256": sha256(BASE_CHECKPOINT),
            },
            "STDE-CDM": {
                "deterministic_head": "spatiotemporal expert joint forecast",
                "full_scenarios": "0.6 distribution expert + 0.4 spatiotemporal expert",
                "checkpoint": str(ST_CHECKPOINT.resolve()),
                "sha256": sha256(ST_CHECKPOINT),
            },
        },
        "data": {
            "path": str(DATA_FILE.resolve()),
            "sha256": sha256(DATA_FILE),
            "test_observations_used_only_for_final_backtest": True,
        },
        "scenario_protocol": {
            "initial_count": args.selected_count,
            "candidate_count": args.candidate_count,
            "selected_count": args.selected_count,
            "maximum_witness_count": args.max_witnesses,
            "validation_count": args.validation_count,
            "initial_seed_rule": "20260725 + day_index",
            "validation_seed_rule": "30260725 + day_index",
            "candidate_seed_rule": "40260725 + day_index",
            "uniform_core_seed_rule": "50260725 + day_index",
            "paired_reverse_diffusion_noise": True,
            "candidate_validation_independence": True,
        },
        "fica": {
            "epsilon": 0.03,
            "theta": 0.06,
            "wind_share": 0.45,
            "mip_gap": 0.001,
            "threads": 4,
            "time_limit_seconds": 14400,
            "feasibility_tolerance_mw": args.feasibility_tolerance,
        },
        "solve_count": 4 * len(days),
        "restart_policy": (
            "NPZ plus complete JSON commits each case; completed cases skip; "
            "an interrupted active solve is repeated with cached scenarios"
        ),
        "script_sha256": {
            str(DAY_SCRIPT): sha256(DAY_SCRIPT),
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--days", type=int, nargs="*")
    parser.add_argument("--candidate-count", type=int, default=6000)
    parser.add_argument("--validation-count", type=int, default=5000)
    parser.add_argument("--selected-count", type=int, default=200)
    parser.add_argument("--max-witnesses", type=int, default=100)
    parser.add_argument("--feasibility-tolerance", type=float, default=1e-4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    days = list(range(50)) if args.days is None else sorted(set(args.days))
    if not days or min(days) < 0 or max(days) >= 50:
        raise ValueError("days must be nonempty and within locked TEST indices 0..49")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_file = output / "experiment_manifest.json"
    proposed = manifest(output, days, args)
    if manifest_file.is_file():
        existing = json.loads(manifest_file.read_text(encoding="utf-8"))
        # Immutable scientific settings must match on resume.
        for key in ("locked_test_days", "model_seed", "models", "data", "scenario_protocol", "fica"):
            if existing.get(key) != proposed.get(key):
                raise RuntimeError(f"manifest mismatch on resume: {key}")
    else:
        atomic_json(manifest_file, proposed)

    atomic_json(
        output / "runner_state.json",
        {
            "state": "validated" if args.dry_run else "running",
            "pid": os.getpid(),
            "started_utc": utc_now(),
            "days": days,
        },
    )
    print(f"[formal] output={output}", flush=True)
    print(f"[formal] days={len(days)} expected_solves={4 * len(days)}", flush=True)
    if args.dry_run:
        aggregate(output, days)
        print("[formal] dry-run validation complete", flush=True)
        return

    for sequence, day in enumerate(days, 1):
        day_output = output / "days" / f"day_{day:02d}"
        command = [
            sys.executable, str(DAY_SCRIPT),
            "--day", str(day),
            "--candidate-count", str(args.candidate_count),
            "--validation-count", str(args.validation_count),
            "--selected-count", str(args.selected_count),
            "--max-witnesses", str(args.max_witnesses),
            "--stde-center", "spatiotemporal",
            "--feasibility-tolerance", str(args.feasibility_tolerance),
            "--output-dir", str(day_output),
        ]
        print(f"\n[formal] day {sequence}/{len(days)} index={day:02d}", flush=True)
        subprocess.run(command, check=True)
        progress = aggregate(output, days)
        atomic_json(
            output / "runner_state.json",
            {
                "state": "running",
                "pid": os.getpid(),
                "updated_utc": utc_now(),
                "last_completed_day": day,
                "completed_day_count": progress["completed_day_count"],
                "completed_case_count": progress["completed_case_count"],
                "expected_case_count": progress["expected_case_count"],
            },
        )

    progress = aggregate(output, days)
    atomic_json(
        output / "runner_state.json",
        {
            "state": "complete",
            "pid": os.getpid(),
            "completed_utc": utc_now(),
            "completed_day_count": progress["completed_day_count"],
            "completed_case_count": progress["completed_case_count"],
        },
    )
    atomic_json(output / "final_summary.json", progress)
    print("[formal] all requested days complete", flush=True)


if __name__ == "__main__":
    main()
