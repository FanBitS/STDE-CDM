"""Spatio-temporal joint conditional diffusion for multi-farm scenarios."""
from __future__ import annotations

import torch
from torch import nn
from cldm.model import StepEmbedding
from .joint_cldm import JointDenoiser


class SpatioTemporalBlock(nn.Module):
    """Alternates temporal dilated convolution and cross-farm attention."""
    def __init__(self, channels: int, dilation: int, heads: int = 4):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.SiLU(), nn.Conv1d(channels, channels, 1),
        )
        self.temporal_norm = nn.LayerNorm(channels)
        self.spatial = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.spatial_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, 2 * channels), nn.SiLU(),
                                 nn.Linear(2 * channels, channels))
        self.ffn_norm = nn.LayerNorm(channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: batch, hour, farm, channel
        batch, hours, farms, channels = features.shape
        temporal = features.permute(0, 2, 3, 1).reshape(batch * farms, channels, hours)
        temporal = self.temporal(temporal).reshape(batch, farms, channels, hours)
        temporal = temporal.permute(0, 3, 1, 2)
        features = self.temporal_norm(features + temporal)
        spatial = features.reshape(batch * hours, farms, channels)
        attended, _ = self.spatial(spatial, spatial, spatial, need_weights=False)
        spatial = self.spatial_norm(spatial + attended)
        spatial = self.ffn_norm(spatial + self.ffn(spatial))
        return spatial.reshape(batch, hours, farms, channels)


class JointSpatioTemporalEncoder(nn.Module):
    def __init__(self, nwp_features=4, channels=64, layers=3):
        super().__init__()
        self.input = nn.Linear(nwp_features, channels)
        self.farm_embedding = nn.Parameter(torch.randn(1, 1, 5, channels) * 0.02)
        self.blocks = nn.ModuleList(
            SpatioTemporalBlock(channels, 2 ** index) for index in range(layers)
        )
        self.forecast_head = nn.Sequential(nn.Linear(channels, channels), nn.SiLU(),
                                           nn.Linear(channels, 1), nn.Sigmoid())

    def forward(self, nwp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.input(nwp) + self.farm_embedding[:, :, :nwp.shape[2]]
        for block in self.blocks:
            features = block(features)
        return self.forecast_head(features).squeeze(-1), features


class SpatioTemporalDenoiser(nn.Module):
    def __init__(self, farms=5, channels=64, layers=3, steps=50):
        super().__init__()
        self.noisy_input = nn.Linear(1, channels)
        self.forecast_input = nn.Linear(1, channels)
        self.condition_input = nn.Linear(channels, channels)
        self.step_embedding = StepEmbedding(steps, channels)
        self.blocks = nn.ModuleList(
            SpatioTemporalBlock(channels, 2 ** index) for index in range(layers)
        )
        self.output = nn.Sequential(nn.Linear(channels, channels), nn.SiLU(),
                                    nn.Linear(channels, 1))

    def forward(self, noisy: torch.Tensor, step: torch.Tensor,
                forecast: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        step_features = self.step_embedding(step).squeeze(-1)[:, None, None, :]
        features = self.noisy_input(noisy.unsqueeze(-1))
        # Keep the CLDM negative-correlation conditioning, now enriched by the
        # jointly encoded spatio-temporal NWP representation.
        features = features - self.forecast_input(forecast.unsqueeze(-1))
        features = features + self.condition_input(condition) + step_features
        for block in self.blocks:
            features = block(features)
        return self.output(features).squeeze(-1)


class STJCDM(nn.Module):
    """Joint point forecast plus residual diffusion in a shared ST feature space."""
    def __init__(self, farms=5, channels=64, layers=3, steps=50,
                 beta_start=1e-4, beta_end=0.05):
        super().__init__(); self.farms = farms; self.steps = steps
        self.encoder = JointSpatioTemporalEncoder(4, channels, layers)
        self.denoiser = SpatioTemporalDenoiser(farms, channels, layers, steps)
        betas = torch.linspace(beta_start, beta_end, steps); alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, 0)
        previous = torch.cat([torch.ones(1), alpha_bars[:-1]])
        posterior = betas * (1 - previous) / (1 - alpha_bars)
        for name, value in [("betas", betas), ("alphas", alphas),
                            ("alpha_bars", alpha_bars),
                            ("posterior_variance", posterior.clamp_min(1e-20))]:
            self.register_buffer(name, value)

    def forecast(self, nwp):
        return self.encoder(nwp)[0]

    def embedding_loss(self, nwp, target):
        forecast, _ = self.encoder(nwp)
        point = nn.functional.l1_loss(forecast, target)
        aggregate = nn.functional.l1_loss(forecast.mean(2), target.mean(2))
        return point + 0.2 * aggregate, {"point": point, "aggregate": aggregate}

    def diffusion_loss(self, nwp, target, step=None, noise=None):
        with torch.no_grad():
            forecast, condition = self.encoder(nwp)
        clean = target - forecast
        if step is None:
            step = torch.randint(0, self.steps, (len(target),), device=target.device)
        if noise is None: noise = torch.randn_like(clean)
        alpha_bar = self.alpha_bars[step, None, None]
        noisy = alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise
        return nn.functional.mse_loss(
            self.denoiser(noisy, step, forecast, condition), noise
        )

    @torch.no_grad()
    def sample(self, nwp, scenarios, generator=None):
        forecast, condition = self.encoder(nwp); days, hours, farms = forecast.shape
        forecast = forecast[:, None].expand(days, scenarios, hours, farms).reshape(-1, hours, farms)
        condition = condition[:, None].expand(days, scenarios, *condition.shape[1:]).reshape(
            -1, *condition.shape[1:])
        error = torch.randn(forecast.shape, device=nwp.device, dtype=nwp.dtype,
                            generator=generator)
        for index in reversed(range(self.steps)):
            step = torch.full((len(error),), index, device=nwp.device, dtype=torch.long)
            predicted = self.denoiser(error, step, forecast, condition)
            alpha, alpha_bar = self.alphas[index], self.alpha_bars[index]
            error = (error - (1 - alpha) / (1 - alpha_bar).sqrt() * predicted) / alpha.sqrt()
            if index:
                error = error + self.posterior_variance[index].sqrt() * torch.randn(
                    error.shape, device=error.device, dtype=error.dtype, generator=generator)
        return (forecast + error).clamp(0, 1).reshape(days, scenarios, hours, farms)


class STEncoderCLDM(STJCDM):
    """Ablation: joint ST encoder with the proven Joint-CLDM denoiser."""
    def __init__(self, farms=5, channels=64, layers=3, steps=50,
                 beta_start=1e-4, beta_end=0.05):
        super().__init__(farms, channels, layers, steps, beta_start, beta_end)
        self.denoiser = JointDenoiser(farms, channels, layers, steps)

    def diffusion_loss(self, nwp, target, step=None, noise=None):
        with torch.no_grad(): forecast, _ = self.encoder(nwp)
        clean = target - forecast
        if step is None:
            step = torch.randint(0, self.steps, (len(target),), device=target.device)
        if noise is None: noise = torch.randn_like(clean)
        alpha_bar = self.alpha_bars[step, None, None]
        noisy = alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise
        return nn.functional.mse_loss(self.denoiser(noisy, step, forecast), noise)

    @torch.no_grad()
    def sample(self, nwp, scenarios, generator=None):
        forecast, _ = self.encoder(nwp); days, hours, farms = forecast.shape
        forecast = forecast[:, None].expand(days, scenarios, hours, farms).reshape(-1, hours, farms)
        error = torch.randn(forecast.shape, device=nwp.device, dtype=nwp.dtype,
                            generator=generator)
        for index in reversed(range(self.steps)):
            step = torch.full((len(error),), index, device=nwp.device, dtype=torch.long)
            predicted = self.denoiser(error, step, forecast)
            alpha, alpha_bar = self.alphas[index], self.alpha_bars[index]
            error = (error - (1 - alpha) / (1 - alpha_bar).sqrt() * predicted) / alpha.sqrt()
            if index:
                error = error + self.posterior_variance[index].sqrt() * torch.randn(
                    error.shape, device=error.device, dtype=error.dtype, generator=generator)
        return (forecast + error).clamp(0, 1).reshape(days, scenarios, hours, farms)


class RampDomainSTJCDM(STJCDM):
    """Diffuses an invertible anchor-plus-increments representation of errors."""
    @staticmethod
    def to_ramp_domain(values: torch.Tensor) -> torch.Tensor:
        return torch.cat([values[:, :1], torch.diff(values, dim=1)], dim=1)

    @staticmethod
    def from_ramp_domain(values: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(values, dim=1)

    def diffusion_loss(self, nwp, target, step=None, noise=None):
        with torch.no_grad(): forecast, condition = self.encoder(nwp)
        clean = self.to_ramp_domain(target - forecast)
        forecast_ramp = self.to_ramp_domain(forecast)
        if step is None:
            step = torch.randint(0, self.steps, (len(target),), device=target.device)
        if noise is None: noise = torch.randn_like(clean)
        alpha_bar = self.alpha_bars[step, None, None]
        noisy = alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise
        return nn.functional.mse_loss(
            self.denoiser(noisy, step, forecast_ramp, condition), noise
        )

    @torch.no_grad()
    def sample(self, nwp, scenarios, generator=None):
        forecast, condition = self.encoder(nwp); days, hours, farms = forecast.shape
        forecast = forecast[:, None].expand(days, scenarios, hours, farms).reshape(-1, hours, farms)
        forecast_ramp = self.to_ramp_domain(forecast)
        condition = condition[:, None].expand(days, scenarios, *condition.shape[1:]).reshape(
            -1, *condition.shape[1:])
        representation = torch.randn(forecast.shape, device=nwp.device, dtype=nwp.dtype,
                                     generator=generator)
        for index in reversed(range(self.steps)):
            step = torch.full((len(representation),), index, device=nwp.device, dtype=torch.long)
            predicted = self.denoiser(representation, step, forecast_ramp, condition)
            alpha, alpha_bar = self.alphas[index], self.alpha_bars[index]
            representation = (representation - (1 - alpha) / (1 - alpha_bar).sqrt()
                              * predicted) / alpha.sqrt()
            if index:
                representation = representation + self.posterior_variance[index].sqrt() * torch.randn(
                    representation.shape, device=representation.device,
                    dtype=representation.dtype, generator=generator)
        error = self.from_ramp_domain(representation)
        return (forecast + error).clamp(0, 1).reshape(days, scenarios, hours, farms)
