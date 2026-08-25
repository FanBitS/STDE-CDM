from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


METRICS = ["MAE", "RMSE", "CRPS", "PS", "ES", "VS"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--reference", default="distribution_expert")
    parser.add_argument("--candidate", default="fused")
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    summary = {}
    for metric in METRICS:
        reference = np.asarray(
            [row[args.reference][metric] for row in result["rows"]]
        )
        candidate = np.asarray(
            [row[args.candidate][metric] for row in result["rows"]]
        )
        summary[metric] = {
            "better_seeds": int(np.sum(candidate < reference)),
            "paired_t_p_one_sided": float(
                ttest_rel(candidate, reference, alternative="less").pvalue
            ),
            "wilcoxon_p_one_sided": float(
                wilcoxon(candidate, reference, alternative="less").pvalue
            ),
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
