from pathlib import Path
import sys

import torch

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / "src")]
from stde_cdm import build_joint_umnn


def test_joint_umnn_forward_shape():
    model = build_joint_umnn(target_size=6, condition_size=8)
    likelihood, latent = model.compute_ll(torch.randn(2, 6), torch.randn(2, 8))
    assert likelihood.shape == (2,)
    assert latent.shape == (2, 6)
    assert torch.isfinite(likelihood).all()
