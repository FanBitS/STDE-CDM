from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cldm import CLDM, CLDMConfig  # noqa: E402
from cldm.metrics import scenario_metrics  # noqa: E402
from stde_cdm import JointCLDM, load_joint  # noqa: E402


def quantile_map_fit(
    source: np.ndarray, target: np.ndarray, grid_size: int = 1001
) -> tuple[np.ndarray, np.ndarray]:
    quantiles = np.linspace(0, 1, grid_size)
    return np.quantile(source, quantiles), np.quantile(target, quantiles)


def quantile_map(
    values: np.ndarray, source_quantiles: np.ndarray, target_quantiles: np.ndarray
) -> np.ndarray:
    return np.interp(values, source_quantiles, target_quantiles).astype(np.float32)


def metrics(scenarios: np.ndarray, observations: np.ndarray) -> dict[str, float]:
    result = scenario_metrics(
        scenarios.reshape(len(observations), scenarios.shape[1], -1),
        observations.reshape(len(observations), -1),
    )
    scenario_correlation = np.corrcoef(scenarios.reshape(-1, 5), rowvar=False)
    observed_correlation = np.corrcoef(observations.reshape(-1, 5), rowvar=False)
    result["spatial_corr_error"] = float(
        np.linalg.norm(scenario_correlation - observed_correlation)
    )
    return result


def joint_checkpoint(seed: int) -> Path:
    return (
        ROOT
        / "artifacts"
        / "checkpoints"
        / f"joint_cldm_seed{seed}.pt"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only marginal calibration audit for Joint CLDM"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--scenarios", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "metrics" / "joint_cldm_calibration_vs.json",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_joint(ROOT / "data" / "wind_data_all_zone.csv")
    calibration_days = np.arange(25)
    evaluation_days = np.arange(25, 50)
    rows = []
    independent_models = []

    for zone in range(5):
        checkpoint_path = (
            ROOT
            / "artifacts"
            / "checkpoints"
            / "single_site_cldm"
            / f"cldm_zone{zone + 1}.pt"
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model = CLDM(CLDMConfig(**checkpoint["config"])).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        independent_models.append(model)

    for seed in args.seeds:
        checkpoint = torch.load(
            joint_checkpoint(seed), map_location=device, weights_only=False
        )
        model = JointCLDM().to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        torch.manual_seed(3000 + seed)
        calibration_scenarios = model.sample(
            torch.from_numpy(data.x_validation[calibration_days]).to(device),
            args.scenarios,
        ).cpu().numpy()
        evaluation_scenarios = model.sample(
            torch.from_numpy(data.x_validation[evaluation_days]).to(device),
            args.scenarios,
        ).cpu().numpy()
        mapped = evaluation_scenarios.copy()
        for zone in range(5):
            source_quantiles, target_quantiles = quantile_map_fit(
                calibration_scenarios[:, :, :, zone].reshape(-1),
                data.y_validation[calibration_days, :, zone].reshape(-1),
            )
            mapped[:, :, :, zone] = quantile_map(
                evaluation_scenarios[:, :, :, zone],
                source_quantiles,
                target_quantiles,
            )

        farms = []
        for zone, independent in enumerate(independent_models):
            torch.manual_seed(4000 + seed * 10 + zone)
            farms.append(
                independent.sample(
                    torch.from_numpy(
                        data.x_validation[evaluation_days, :, zone, :]
                    ).to(device),
                    args.scenarios,
                ).cpu().numpy()
            )
        independent_scenarios = np.stack(farms, axis=-1)
        rows.append(
            {
                "seed": seed,
                "independent": metrics(
                    independent_scenarios, data.y_validation[evaluation_days]
                ),
                "raw": metrics(
                    evaluation_scenarios, data.y_validation[evaluation_days]
                ),
                "calibrated": metrics(mapped, data.y_validation[evaluation_days]),
            }
        )

    summary = {}
    for key in ["CRPS", "ES", "VS", "spatial_corr_error"]:
        raw = np.asarray([row["raw"][key] for row in rows])
        calibrated = np.asarray([row["calibrated"][key] for row in rows])
        summary[key] = {
            "raw_mean": float(raw.mean()),
            "calibrated_mean": float(calibrated.mean()),
            "relative_change_percent": float(
                100 * (calibrated.mean() - raw.mean()) / raw.mean()
            ),
        }
    output = {
        "split": "VS first 25 calibration / last 25 evaluation",
        "seeds": args.seeds,
        "scenarios_per_day": args.scenarios,
        "rows": rows,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
