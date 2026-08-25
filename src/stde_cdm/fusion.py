"""Paired-noise trajectory fusion used by STDE-CDM."""

from __future__ import annotations

import torch

from .joint_cldm import JointCLDM
from .st_jcdm import STJCDM


@torch.no_grad()
def sample_stde_cdm(
    distribution_expert: JointCLDM,
    spatiotemporal_expert: STJCDM,
    nwp: torch.Tensor,
    scenarios: int,
    *,
    spatiotemporal_weight: float = 0.4,
    seed: int = 0,
) -> torch.Tensor:
    """Generate synchronized STDE-CDM scenarios with paired Gaussian noise.

    Both experts are reset to the same seed before sampling. Because their
    residual arrays have identical shapes and use the same diffusion schedule,
    corresponding scenario indices receive the same initial Gaussian state and
    the same indexed reverse-process innovations. The returned tensor has shape
    ``[days, scenarios, hours, sites]``.
    """
    if not 0.0 <= spatiotemporal_weight <= 1.0:
        raise ValueError("spatiotemporal_weight must lie in [0, 1]")
    if scenarios < 1:
        raise ValueError("scenarios must be positive")

    device = nwp.device
    devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        distribution = distribution_expert.sample(nwp, scenarios)

    generator = torch.Generator(device=device).manual_seed(seed)
    spatiotemporal = spatiotemporal_expert.sample(
        nwp,
        scenarios,
        generator=generator,
    )
    return (
        (1.0 - spatiotemporal_weight) * distribution
        + spatiotemporal_weight * spatiotemporal
    )
