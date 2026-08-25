"""Conditional DDPM benchmark jointly generating five wind farms.

Following the CLDM paper's DDPM benchmark, this model uses the same diffusion
schedule and denoising backbone but directly diffuses wind power.  It has no
deterministic embedding network and no forecast-error latent conversion.
"""
from __future__ import annotations

import torch
from torch import nn
from cldm.model import DilatedResidualBlock, StepEmbedding


class JointDDPMDenoiser(nn.Module):
    def __init__(self, farms=5, nwp_features=4, channels=64, layers=3, steps=50):
        super().__init__()
        self.farms = farms
        self.noisy_input = nn.Conv1d(farms, channels, 1)
        self.condition_input = nn.Conv1d(farms * nwp_features, channels, 1)
        self.step_embedding = StepEmbedding(steps, channels)
        self.blocks = nn.ModuleList(
            DilatedResidualBlock(channels, 2 ** index) for index in range(layers)
        )
        self.output = nn.Sequential(
            nn.LeakyReLU(0.2), nn.Conv1d(channels, channels, 1),
            nn.LeakyReLU(0.2), nn.Conv1d(channels, farms, 1),
        )

    def forward(self, noisy: torch.Tensor, step: torch.Tensor,
                nwp: torch.Tensor) -> torch.Tensor:
        # noisy: (batch, hour, farm); nwp: (batch, hour, farm, feature)
        condition = nwp.reshape(len(nwp), nwp.shape[1], -1).transpose(1, 2)
        hidden = self.noisy_input(noisy.transpose(1, 2))
        hidden = hidden - self.condition_input(condition)
        hidden = hidden + self.step_embedding(step)
        skips = []
        for block in self.blocks:
            hidden, skip = block(hidden)
            skips.append(skip)
        return self.output(torch.stack(skips).sum(0)).transpose(1, 2)


class JointDDPM(nn.Module):
    def __init__(self, farms=5, nwp_features=4, channels=64, layers=3,
                 steps=50, beta_start=1e-4, beta_end=0.05):
        super().__init__()
        self.farms = farms
        self.steps = steps
        self.denoiser = JointDDPMDenoiser(
            farms, nwp_features, channels, layers, steps
        )
        betas = torch.linspace(beta_start, beta_end, steps)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, 0)
        previous = torch.cat([torch.ones(1), alpha_bars[:-1]])
        posterior_variance = betas * (1 - previous) / (1 - alpha_bars)
        for name, value in [
            ("betas", betas), ("alphas", alphas),
            ("alpha_bars", alpha_bars),
            ("posterior_variance", posterior_variance.clamp_min(1e-20)),
        ]:
            self.register_buffer(name, value)

    def loss(self, nwp: torch.Tensor, target: torch.Tensor,
             step: torch.Tensor | None = None,
             noise: torch.Tensor | None = None) -> torch.Tensor:
        if step is None:
            step = torch.randint(0, self.steps, (len(target),), device=target.device)
        if noise is None:
            noise = torch.randn_like(target)
        alpha_bar = self.alpha_bars[step, None, None]
        noisy = alpha_bar.sqrt() * target + (1 - alpha_bar).sqrt() * noise
        return nn.functional.mse_loss(self.denoiser(noisy, step, nwp), noise)

    @torch.no_grad()
    def sample(self, nwp: torch.Tensor, scenarios: int,
               generator: torch.Generator | None = None) -> torch.Tensor:
        days, hours = nwp.shape[:2]
        condition = nwp[:, None].expand(days, scenarios, *nwp.shape[1:]).reshape(
            days * scenarios, *nwp.shape[1:]
        )
        generated = torch.randn(
            days * scenarios, hours, self.farms, device=nwp.device,
            dtype=nwp.dtype, generator=generator,
        )
        for index in reversed(range(self.steps)):
            step = torch.full((len(generated),), index, device=nwp.device,
                              dtype=torch.long)
            predicted_noise = self.denoiser(generated, step, condition)
            alpha, alpha_bar = self.alphas[index], self.alpha_bars[index]
            generated = (generated - (1 - alpha) / (1 - alpha_bar).sqrt()
                         * predicted_noise) / alpha.sqrt()
            if index:
                generated = generated + self.posterior_variance[index].sqrt() * torch.randn(
                    generated.shape, device=generated.device, dtype=generated.dtype,
                    generator=generator,
                )
        return generated.clamp(0, 1).reshape(days, scenarios, hours, self.farms)
