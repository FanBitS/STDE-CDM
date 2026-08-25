from pathlib import Path
import sys

import torch

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / "src")]
from stde_cdm import JointDDPM


def test_joint_ddpm_loss_and_sample():
    model = JointDDPM(channels=8, layers=2, steps=4)
    nwp = torch.randn(2, 24, 5, 4)
    target = torch.rand(2, 24, 5)
    assert torch.isfinite(model.loss(nwp, target))
    scenarios = model.sample(nwp, 3)
    assert scenarios.shape == (2, 3, 24, 5)
    assert scenarios.min() >= 0 and scenarios.max() <= 1
