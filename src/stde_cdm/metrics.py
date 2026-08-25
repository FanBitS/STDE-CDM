from __future__ import annotations

import numpy as np


def ramp_metrics(scenarios: np.ndarray, observations: np.ndarray) -> dict[str, float]:
    """Distributional ramp metrics for (day, scenario, hour) trajectories."""
    generated = np.diff(scenarios, axis=2)
    observed = np.diff(observations, axis=1)
    generated_flat = generated.reshape(-1)
    observed_flat = observed.reshape(-1)
    quantiles = np.asarray([0.01, 0.05, 0.50, 0.95, 0.99])
    generated_q = np.quantile(generated_flat, quantiles, method="inverted_cdf")
    observed_q = np.quantile(observed_flat, quantiles, method="inverted_cdf")

    # Empirical 1-Wasserstein distance via equal-size quantile grids.
    grid = np.linspace(0.001, 0.999, 999)
    wasserstein = np.mean(np.abs(
        np.quantile(generated_flat, grid, method="inverted_cdf")
        - np.quantile(observed_flat, grid, method="inverted_cdf")
    ))
    return {
        "ramp_wasserstein": float(wasserstein),
        "ramp_quantile_mae": float(np.mean(np.abs(generated_q - observed_q))),
        "ramp_q01_error": float(abs(generated_q[0] - observed_q[0])),
        "ramp_q05_error": float(abs(generated_q[1] - observed_q[1])),
        "ramp_q95_error": float(abs(generated_q[3] - observed_q[3])),
        "ramp_q99_error": float(abs(generated_q[4] - observed_q[4])),
        "generated_ramp_std": float(generated_flat.std()),
        "observed_ramp_std": float(observed_flat.std()),
    }
