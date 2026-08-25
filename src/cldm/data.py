from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class WindDataSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_dates: np.ndarray
    train_dates: np.ndarray
    validation_dates: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray


def _daily_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert the competition's 01:00--00:00 records into complete days."""
    features = ["U10", "V10", "U100", "V100"]
    if len(frame) % 24:
        raise ValueError(f"zone has {len(frame)} hourly rows, not a multiple of 24")
    n_days = len(frame) // 24
    x = frame[features].to_numpy(np.float32).reshape(n_days, 24, len(features))
    y = frame["TARGETVAR"].to_numpy(np.float32).reshape(n_days, 24)
    dates = frame.index.to_numpy().reshape(n_days, 24)[:, 0]
    return x, y, dates


def load_gefcom_zone(
    path: str | Path,
    zone: int = 1,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> WindDataSplits:
    """Load the chronological 2012/2013 split used by the CLDM paper."""
    if not 1 <= zone <= 10:
        raise ValueError("zone must be in [1, 10]")
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame = frame.loc[frame[f"ZONE_{zone}"] == 1].sort_index()
    x, y, dates = _daily_arrays(frame)

    # Each competition day starts at 01:00. The first 366 complete blocks are
    # 2012 (leap year), and the remaining 365 are 2013.
    if len(x) != 731:
        raise ValueError(f"expected 731 complete days, found {len(x)}")
    x_2012, y_2012 = x[:366], y[:366]
    x_test, y_test, test_dates = x[366:], y[366:], dates[366:]

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x_2012))
    n_validation = int(round(validation_fraction * len(x_2012)))
    validation_ids, train_ids = order[:n_validation], order[n_validation:]

    x_train_raw = x_2012[train_ids]
    x_mean = x_train_raw.mean(axis=(0, 1), keepdims=True)
    x_std = x_train_raw.std(axis=(0, 1), keepdims=True)
    x_std = np.maximum(x_std, 1e-6)

    def standardize(values: np.ndarray) -> np.ndarray:
        return ((values - x_mean) / x_std).astype(np.float32)

    return WindDataSplits(
        x_train=standardize(x_2012[train_ids]),
        y_train=y_2012[train_ids],
        x_validation=standardize(x_2012[validation_ids]),
        y_validation=y_2012[validation_ids],
        x_test=standardize(x_test),
        y_test=y_test,
        test_dates=test_dates,
        train_dates=dates[:366][train_ids],
        validation_dates=dates[:366][validation_ids],
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
    )


def load_gefcom_zone_unified(
    path: str | Path,
    zone: int = 1,
    test_size: int = 50,
    seed: int = 0,
) -> WindDataSplits:
    """Load the exact random LS/VS/TEST split used by the existing baselines."""
    if not 1 <= zone <= 10:
        raise ValueError("zone must be in [1, 10]")
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame = frame.loc[frame[f"ZONE_{zone}"] == 1].sort_index()
    x, y, dates = _daily_arrays(frame)
    indices = np.arange(len(x))
    train_validation, test_ids = train_test_split(
        indices, test_size=test_size, random_state=seed, shuffle=True
    )
    train_ids, validation_ids = train_test_split(
        train_validation, test_size=test_size, random_state=seed, shuffle=True
    )

    x_mean = x[train_ids].mean(axis=(0, 1), keepdims=True)
    x_std = np.maximum(x[train_ids].std(axis=(0, 1), keepdims=True), 1e-6)

    def standardize(values: np.ndarray) -> np.ndarray:
        return ((values - x_mean) / x_std).astype(np.float32)

    return WindDataSplits(
        x_train=standardize(x[train_ids]), y_train=y[train_ids],
        x_validation=standardize(x[validation_ids]), y_validation=y[validation_ids],
        x_test=standardize(x[test_ids]), y_test=y[test_ids],
        test_dates=dates[test_ids], train_dates=dates[train_ids],
        validation_dates=dates[validation_ids],
        x_mean=x_mean.astype(np.float32), x_std=x_std.astype(np.float32),
    )
