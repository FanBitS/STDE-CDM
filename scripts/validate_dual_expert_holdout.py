from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cldm.metrics import scenario_metrics  # noqa: E402
from stde_cdm import JointCLDM, STJCDM, load_joint  # noqa: E402


METRICS = ["MAE", "RMSE", "CRPS", "PS", "ES", "VS"]


def metrics(scenarios: np.ndarray, observations: np.ndarray) -> dict[str, float]:
    return scenario_metrics(
        scenarios.reshape(len(observations), scenarios.shape[1], -1),
        observations.reshape(len(observations), -1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the dual-expert fusion weight on held-out validation days"
    )
    parser.add_argument("--scenarios", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "metrics" / "dual_expert_holdout_vs.json",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_joint(ROOT / "data" / "wind_data_all_zone.csv")
    condition = torch.from_numpy(data.x_validation).to(device)

    base_checkpoint = torch.load(
        ROOT / "artifacts" / "checkpoints" / "joint_cldm_seed0.pt",
        map_location=device,
        weights_only=False,
    )
    distribution = JointCLDM().to(device)
    distribution.load_state_dict(base_checkpoint["state_dict"])
    distribution.eval()

    st_checkpoint = torch.load(
        ROOT
        / "artifacts"
        / "checkpoints"
        / "stde_spatiotemporal_seed0.pt",
        map_location=device,
        weights_only=False,
    )
    spatiotemporal = STJCDM(**st_checkpoint["config"]).to(device)
    spatiotemporal.load_state_dict(st_checkpoint["state_dict"])
    spatiotemporal.eval()

    paired_seed = 41000
    torch.manual_seed(paired_seed)
    distribution_scenarios = distribution.sample(
        condition, args.scenarios
    ).cpu().numpy()
    generator = torch.Generator(device=device).manual_seed(paired_seed)
    spatiotemporal_scenarios = spatiotemporal.sample(
        condition, args.scenarios, generator
    ).cpu().numpy()

    calibration_days = np.arange(25)
    holdout_days = np.arange(25, 50)
    weights = np.linspace(0, 1, 21)
    base_calibration = metrics(
        distribution_scenarios[calibration_days],
        data.y_validation[calibration_days],
    )
    rows = []
    for weight in weights:
        candidate = (
            (1 - weight) * distribution_scenarios[calibration_days]
            + weight * spatiotemporal_scenarios[calibration_days]
        )
        result = metrics(candidate, data.y_validation[calibration_days])
        score = float(np.mean([result[key] / base_calibration[key] for key in METRICS]))
        rows.append({"weight": float(weight), "score": score, **result})

    selected = min(rows, key=lambda row: row["score"])
    weight = selected["weight"]
    base_holdout = metrics(
        distribution_scenarios[holdout_days], data.y_validation[holdout_days]
    )
    fused_holdout = metrics(
        (1 - weight) * distribution_scenarios[holdout_days]
        + weight * spatiotemporal_scenarios[holdout_days],
        data.y_validation[holdout_days],
    )
    changes = {
        key: 100 * (fused_holdout[key] - base_holdout[key]) / base_holdout[key]
        for key in base_holdout
    }
    output = {
        "protocol": (
            "VS days 0:25 calibration using the six reported metrics; "
            "days 25:50 held out"
        ),
        "selection_metrics": METRICS,
        "scenarios_per_day": args.scenarios,
        "selected_weight_st": weight,
        "calibration": rows,
        "heldout_base": base_holdout,
        "heldout_fused": fused_holdout,
        "heldout_relative_change_percent": changes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: output[key]
                for key in (
                    "selected_weight_st",
                    "heldout_base",
                    "heldout_fused",
                    "heldout_relative_change_percent",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
