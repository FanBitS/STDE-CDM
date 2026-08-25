from pathlib import Path
import sys

import torch

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / "src")]
from stde_cdm import JointVAE


def test_joint_vae_loss_and_sample():
    model = JointVAE(latent_size=4, hidden_size=16)
    condition = torch.randn(2, 24, 5, 4)
    target = torch.rand(2, 24, 5)
    loss, components = model.loss(condition, target)
    assert torch.isfinite(loss)
    assert set(components) == {"reconstruction", "kl"}
    generated = model.sample(condition, 3)
    assert generated.shape == (2, 3, 24, 5)
