#!/usr/bin/env python3
"""Build the single authoritative locked-TEST record used by the paper.

The historical all-model evaluation predates the final dual-expert lock.  This
script retains its four non-diffusion baselines and combines them with the
paired Joint CLDM and final STDE CDM values from the frozen dual-expert run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
BASELINE_SOURCE = METRICS / "all_joint_models_locked_test_10seed.json"
DUAL_SOURCE = METRICS / "dual_expert_locked_test_weight_0p4.json"
OUTPUT = METRICS / "paper_locked_test_10seed.json"

METRIC_NAMES = ["MAE", "RMSE", "CRPS", "PS", "ES", "VS"]
BASELINE_MODELS = ["Joint-WGAN-GP", "Joint-VAE", "Joint-UMNN", "Joint-DDPM"]
PAPER_MODELS = BASELINE_MODELS + ["Joint-CLDM", "STDE-CDM"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    baseline = json.loads(BASELINE_SOURCE.read_text())
    dual = json.loads(DUAL_SOURCE.read_text())
    baseline_rows = {int(row["seed"]): row for row in baseline["rows"]}
    dual_rows = {int(row["seed"]): row for row in dual["rows"]}
    seeds = sorted(set(baseline_rows) & set(dual_rows))
    if seeds != list(range(10)):
        raise ValueError(f"Expected matched seeds 0--9, found {seeds}")

    rows = []
    for seed in seeds:
        old = baseline_rows[seed]
        paired = dual_rows[seed]
        old_cldm = old["models"]["Joint-CLDM"]
        final_cldm = paired["distribution_expert"]
        for metric in METRIC_NAMES:
            if not np.isclose(old_cldm[metric], final_cldm[metric], rtol=0, atol=1e-12):
                raise ValueError(
                    f"Joint CLDM source mismatch for seed {seed}, metric {metric}"
                )

        models = {name: old["models"][name] for name in BASELINE_MODELS}
        models["Joint-CLDM"] = final_cldm
        models["STDE-CDM"] = paired["fused"]
        rows.append(
            {
                "seed": seed,
                "paired_sampling_seed": paired["paired_sampling_seed"],
                "models": models,
            }
        )

    summary = {}
    for model in PAPER_MODELS:
        summary[model] = {}
        for metric in METRIC_NAMES:
            values = np.asarray(
                [row["models"][model][metric] for row in rows], dtype=float
            )
            summary[model][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
            }

    payload = {
        "schema_version": 1,
        "protocol": (
            "Final paper comparison: 10 matched training seeds, 50 locked TEST "
            "days and 200 synchronized joint scenarios per day; no TEST-based "
            "selection or failed-seed removal."
        ),
        "source_records": {
            BASELINE_SOURCE.name: sha256(BASELINE_SOURCE),
            DUAL_SOURCE.name: sha256(DUAL_SOURCE),
        },
        "seeds": seeds,
        "locked_test_days": 50,
        "scenarios_per_day": 200,
        "models": PAPER_MODELS,
        "rows": rows,
        "summary": summary,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote={OUTPUT}")


if __name__ == "__main__":
    main()
