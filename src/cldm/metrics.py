from __future__ import annotations

import numpy as np


def scenario_metrics(
    scenarios: np.ndarray,
    observations: np.ndarray,
    point_forecast: np.ndarray | None = None,
) -> dict[str, float]:
    """Paper metrics for scenarios shaped (day, scenario, hour)."""
    point = scenarios.mean(axis=1) if point_forecast is None else point_forecast
    error = point - observations
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))

    absolute_observation = np.abs(scenarios - observations[:, None, :]).mean(axis=1)
    sorted_scenarios = np.sort(scenarios, axis=1)
    m = scenarios.shape[1]
    coefficients = (2 * np.arange(1, m + 1) - m - 1).reshape(1, m, 1)
    pair_term = (sorted_scenarios * coefficients).sum(axis=1) / (m * m)
    crps = np.mean(absolute_observation - pair_term)

    quantiles = np.arange(0.05, 1.0, 0.05)
    predicted_quantiles = np.quantile(scenarios, quantiles, axis=1)
    residual = observations[None, :, :] - predicted_quantiles
    pinball = np.maximum(quantiles[:, None, None] * residual,
                         (quantiles[:, None, None] - 1.0) * residual).mean()

    norm_to_observation = np.linalg.norm(scenarios - observations[:, None, :], axis=2).mean(axis=1)
    # ES second term computed day-by-day to bound memory at 200 scenarios.
    es_pair = []
    for day in scenarios:
        es_pair.append(np.linalg.norm(day[:, None, :] - day[None, :, :], axis=2).mean() / 2.0)
    energy_score = np.mean(norm_to_observation - np.asarray(es_pair))

    observed_differences = np.abs(observations[:, :, None] - observations[:, None, :]) ** 0.5
    generated_differences = np.abs(
        scenarios[:, :, :, None] - scenarios[:, :, None, :]
    ) ** 0.5
    variogram_score = np.mean(
        np.sum((observed_differences - generated_differences.mean(axis=1)) ** 2, axis=(1, 2))
    )
    return {
        "MAE": float(mae), "RMSE": float(rmse), "CRPS": float(crps),
        "PS": float(pinball), "ES": float(energy_score), "VS": float(variogram_score),
    }
