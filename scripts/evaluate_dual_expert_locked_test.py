from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cldm.metrics import scenario_metrics
from stde_cdm import JointCLDM, STJCDM, load_joint


METRICS = ["MAE", "RMSE", "CRPS", "PS", "ES", "VS"]


def evaluate(scenarios: np.ndarray, observations: np.ndarray) -> dict[str, float]:
    return scenario_metrics(
        scenarios.reshape(len(observations), scenarios.shape[1], -1),
        observations.reshape(len(observations), -1),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=float, default=0.4)
    parser.add_argument("--scenarios", type=int, default=200)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Training seeds to evaluate; all paper seeds are packaged.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/metrics/dual_expert_locked_test_weight_0p4.json",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_joint(ROOT / "data/wind_data_all_zone.csv")
    condition = torch.from_numpy(data.x_test).to(device)
    rows: list[dict] = []

    for seed in args.seeds:
        started = time.perf_counter()
        base_path = (
            ROOT / "artifacts" / "checkpoints" / f"joint_cldm_seed{seed}.pt"
        )
        base_checkpoint = torch.load(
            base_path, map_location=device, weights_only=False
        )
        base = JointCLDM().to(device)
        base.load_state_dict(base_checkpoint["state_dict"])
        base.eval()

        st_path = (
            ROOT
            / "artifacts"
            / "checkpoints"
            / f"stde_spatiotemporal_seed{seed}.pt"
        )
        st_checkpoint = torch.load(st_path, map_location=device, weights_only=False)
        st = STJCDM(**st_checkpoint["config"]).to(device)
        st.load_state_dict(st_checkpoint["state_dict"])
        st.eval()

        paired_seed = 85000 + seed
        torch.manual_seed(paired_seed)
        base_scenarios = base.sample(condition, args.scenarios).cpu().numpy()
        generator = torch.Generator(device=device).manual_seed(paired_seed)
        st_scenarios = st.sample(condition, args.scenarios, generator).cpu().numpy()
        fused_scenarios = (
            (1.0 - args.weight) * base_scenarios
            + args.weight * st_scenarios
        )

        row = {
            "seed": seed,
            "paired_sampling_seed": paired_seed,
            "elapsed_seconds": time.perf_counter() - started,
            "distribution_expert": evaluate(base_scenarios, data.y_test),
            "spatiotemporal_expert": evaluate(st_scenarios, data.y_test),
            "fused": evaluate(fused_scenarios, data.y_test),
        }
        rows.append(row)
        print(seed, json.dumps(row), flush=True)

        del base, st
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for model in ["distribution_expert", "spatiotemporal_expert", "fused"]:
        summary[model] = {}
        for metric in METRICS:
            values = np.asarray([row[model][metric] for row in rows])
            summary[model][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1 if len(values) > 1 else 0)),
            }

    relative_change = {
        metric: 100.0
        * (
            summary["fused"][metric]["mean"]
            - summary["distribution_expert"][metric]["mean"]
        )
        / summary["distribution_expert"][metric]["mean"]
        for metric in METRICS
    }
    output = {
        "protocol": (
            "Locked TEST evaluation with paired reverse-process noise; "
            "the fusion weight was selected using the six reported metrics "
            "on the calibration portion of the validation set."
        ),
        "weight_spatiotemporal": args.weight,
        "scenarios_per_day": args.scenarios,
        "seeds": args.seeds,
        "rows": rows,
        "summary": summary,
        "relative_change_vs_distribution_expert_percent": relative_change,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps({"summary": summary, "relative_change": relative_change}, indent=2))


if __name__ == "__main__":
    main()
