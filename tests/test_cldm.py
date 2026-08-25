from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cldm import CLDM, CLDMConfig, load_gefcom_zone, load_gefcom_zone_unified
from cldm.metrics import scenario_metrics


def test_data_split_and_shapes():
    data = load_gefcom_zone(ROOT / "data/wind_data_all_zone.csv")
    assert data.x_train.shape == (293, 24, 4)
    assert data.x_validation.shape == (73, 24, 4)
    assert data.x_test.shape == (365, 24, 4)
    assert data.y_test.shape == (365, 24)
    assert np.allclose(data.x_train.mean((0, 1)), 0, atol=1e-5)


def test_forward_loss_and_sampling():
    config = CLDMConfig(channels=8, embedding_layers=2, denoising_layers=2, diffusion_steps=4)
    model = CLDM(config)
    x = torch.randn(3, 24, 4)
    y = torch.rand(3, 24)
    assert model.forecast(x).shape == y.shape
    loss = model.diffusion_loss(x, y)
    assert loss.ndim == 0 and torch.isfinite(loss)
    samples = model.sample(x, 5)
    assert samples.shape == (3, 5, 24)
    assert samples.min() >= 0 and samples.max() <= 1


def test_unified_split_matches_fica_day_one_without_leakage():
    data = load_gefcom_zone_unified(
        ROOT / "data/wind_data_all_zone.csv"
    )
    assert data.x_train.shape == (631, 24, 4)
    assert data.x_validation.shape == (50, 24, 4)
    assert data.x_test.shape == (50, 24, 4)
    fica_day_one = np.datetime64("2012-07-06T01:00:00")
    assert data.test_dates[1] == fica_day_one
    assert fica_day_one not in data.train_dates
    assert fica_day_one not in data.validation_dates


def test_metrics_are_zero_for_perfect_identical_scenarios():
    observations = np.random.default_rng(0).random((2, 24))
    scenarios = np.repeat(observations[:, None, :], 4, axis=1)
    metrics = scenario_metrics(scenarios, observations)
    for value in metrics.values():
        assert abs(value) < 1e-12
