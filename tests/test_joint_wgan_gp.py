from pathlib import Path
import sys

import torch

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / "src")]
from stde_cdm import JointWGANGenerator, JointWGANCritic


def test_joint_wgan_shapes_and_losses():
    generator = JointWGANGenerator(width=16, layers=1, latent_size=4)
    critic = JointWGANCritic(width=16, layers=1)
    condition = torch.randn(3, 24, 5, 4)
    real = torch.randn(3, 24, 5)
    fake = generator(torch.randn(3, 4), condition)
    loss, parts = critic.loss(fake, real, condition)
    assert fake.shape == (3, 120)
    assert torch.isfinite(loss)
    assert set(parts) == {"wasserstein", "gradient_penalty"}
    assert generator.sample(condition, 2).shape == (3, 2, 24, 5)
