"""Convert GEFCom wind scenario matrices to the FICA wind input format."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HOURS_PER_DAY = 24
DEFAULT_NUM_ZONES = 10
DEFAULT_DAYS_PER_ZONE = 50


@dataclass(frozen=True)
class WindScenarioBundle:
    """One conditional forecast case prepared for the dispatch model."""

    point_forecast_pu: np.ndarray
    train_scenarios_pu: np.ndarray
    test_scenarios_pu: np.ndarray
    train_errors_pu: np.ndarray
    test_errors_pu: np.ndarray
    zone: int
    day: int

    def to_mw(
        self,
        capacity_mw: float,
        horizon: int | None = None,
        start_hour: int = 0,
    ) -> dict[str, np.ndarray]:
        if capacity_mw <= 0:
            raise ValueError("capacity_mw must be positive")
        hours = HOURS_PER_DAY if horizon is None else horizon
        if start_hour < 0 or start_hour + hours > HOURS_PER_DAY:
            raise ValueError("start_hour and horizon must define a window within 24 hours")
        time_slice = slice(start_hour, start_hour + hours)

        return {
            "WT_pred": (self.point_forecast_pu[time_slice] * capacity_mw)[:, None],
            "WT_error_scenarios_train": (
                self.train_errors_pu[:, time_slice] * capacity_mw
            )[:, :, None],
            "WT_error_scenarios_test": (
                self.test_errors_pu[:, time_slice] * capacity_mw
            )[:, :, None],
        }


def load_scenario_matrix(path: str | Path) -> np.ndarray:
    """Load an exported scenario pickle with shape (zone-day-hour, scenario)."""
    scenario_path = Path(path)
    if not scenario_path.is_file():
        raise FileNotFoundError(scenario_path)
    with scenario_path.open("rb") as handle:
        matrix = np.asarray(pickle.load(handle), dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D scenario matrix, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("scenario matrix contains NaN or infinite values")
    return matrix


def reshape_wind_scenarios(
    matrix: np.ndarray,
    num_zones: int = DEFAULT_NUM_ZONES,
    days_per_zone: int = DEFAULT_DAYS_PER_ZONE,
) -> np.ndarray:
    """Return scenarios as (zone, day, hour, scenario)."""
    expected_rows = num_zones * days_per_zone * HOURS_PER_DAY
    if matrix.shape[0] != expected_rows:
        raise ValueError(
            f"expected {expected_rows} rows for {num_zones} zones and "
            f"{days_per_zone} days, got {matrix.shape[0]}"
        )
    return matrix.reshape(num_zones, days_per_zone, HOURS_PER_DAY, matrix.shape[1])


def prepare_wind_case(
    path: str | Path,
    zone: int = 0,
    day: int = 0,
    train_fraction: float = 0.7,
    seed: int = 0,
    point_method: str = "median",
) -> WindScenarioBundle:
    """Select one zone-day and split its conditional scenarios without leakage.

    Zone and day use zero-based indices. The point forecast is estimated from
    training scenarios only; test scenarios never influence the dispatch input.
    """
    matrix = load_scenario_matrix(path)
    shaped = reshape_wind_scenarios(matrix)
    if not 0 <= zone < shaped.shape[0]:
        raise IndexError(f"zone must be in [0, {shaped.shape[0] - 1}]")
    if not 0 <= day < shaped.shape[1]:
        raise IndexError(f"day must be in [0, {shaped.shape[1] - 1}]")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")

    trajectories = shaped[zone, day].T  # (scenario, hour)
    rng = np.random.RandomState(seed)
    order = rng.permutation(trajectories.shape[0])
    split = int(np.floor(train_fraction * trajectories.shape[0]))
    if split < 2 or trajectories.shape[0] - split < 1:
        raise ValueError("scenario split leaves too few training or test trajectories")
    train = trajectories[order[:split]]
    test = trajectories[order[split:]]

    if point_method == "median":
        point = np.median(train, axis=0)
    elif point_method == "mean":
        point = np.mean(train, axis=0)
    else:
        raise ValueError("point_method must be 'median' or 'mean'")

    return WindScenarioBundle(
        point_forecast_pu=point,
        train_scenarios_pu=train,
        test_scenarios_pu=test,
        train_errors_pu=train - point[None, :],
        test_errors_pu=test - point[None, :],
        zone=zone,
        day=day,
    )
