from pathlib import Path
import sys
import torch
R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / "src")]
from stde_cdm import STJCDM, STEncoderCLDM, RampDomainSTJCDM


def test_st_jcdm_shapes():
    model = STJCDM(channels=8, layers=2, steps=4)
    nwp = torch.randn(2, 24, 5, 4); target = torch.rand(2, 24, 5)
    forecast = model.forecast(nwp)
    embedding, parts = model.embedding_loss(nwp, target)
    assert forecast.shape == target.shape and torch.isfinite(embedding)
    assert set(parts) == {"point", "aggregate"}
    assert torch.isfinite(model.diffusion_loss(nwp, target))
    assert model.sample(nwp, 3).shape == (2, 3, 24, 5)


def test_st_encoder_cldm_shapes():
    model = STEncoderCLDM(channels=8, layers=2, steps=4)
    nwp = torch.randn(2, 24, 5, 4); target = torch.rand(2, 24, 5)
    assert torch.isfinite(model.diffusion_loss(nwp, target))
    assert model.sample(nwp, 2).shape == (2, 2, 24, 5)


def test_ramp_domain_is_invertible_and_samples():
    model = RampDomainSTJCDM(channels=8, layers=2, steps=4)
    values = torch.randn(2, 24, 5)
    assert torch.allclose(model.from_ramp_domain(model.to_ramp_domain(values)), values,
                          atol=1e-6)
    nwp = torch.randn(2, 24, 5, 4); target = torch.rand(2, 24, 5)
    assert torch.isfinite(model.diffusion_loss(nwp, target))
    assert model.sample(nwp, 2).shape == (2, 2, 24, 5)
