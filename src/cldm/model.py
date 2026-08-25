from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class CLDMConfig:
    nwp_features: int = 4
    horizon: int = 24
    channels: int = 64
    embedding_layers: int = 3
    denoising_layers: int = 3
    diffusion_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.05


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 1),
        )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.net(x)
        return self.activation(x + skip), skip


class EmbeddingNetwork(nn.Module):
    """Paper Fig. 4: dilated convolutions with residual and skip paths."""

    def __init__(self, nwp_features: int, channels: int, layers: int):
        super().__init__()
        self.input = nn.Conv1d(nwp_features, channels, 1)
        self.blocks = nn.ModuleList(
            DilatedResidualBlock(channels, 2**i) for i in range(layers)
        )
        self.output = nn.Sequential(
            nn.LeakyReLU(0.2), nn.Conv1d(channels, channels, 1),
            nn.LeakyReLU(0.2), nn.Conv1d(channels, 1, 1), nn.Sigmoid()
        )

    def forward(self, nwp: torch.Tensor) -> torch.Tensor:
        hidden = self.input(nwp.transpose(1, 2))
        skips = []
        for block in self.blocks:
            hidden, skip = block(hidden)
            skips.append(skip)
        return self.output(torch.stack(skips).sum(0)).squeeze(1)


class StepEmbedding(nn.Module):
    def __init__(self, steps: int, channels: int):
        super().__init__()
        positions = torch.arange(steps, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            -torch.arange(0, channels, 2, dtype=torch.float32)
            * (torch.log(torch.tensor(10000.0)) / channels)
        )
        table = torch.zeros(steps, channels)
        table[:, 0::2] = torch.sin(positions * frequencies)
        table[:, 1::2] = torch.cos(positions * frequencies[: table[:, 1::2].shape[1]])
        self.register_buffer("table", table)
        self.projection = nn.Sequential(
            nn.Linear(channels, channels), nn.LeakyReLU(0.2),
            nn.Linear(channels, channels)
        )

    def forward(self, step: torch.Tensor) -> torch.Tensor:
        return self.projection(self.table[step]).unsqueeze(-1)


class DenoisingNetwork(nn.Module):
    """Paper Fig. 5: error, deterministic forecast and step-conditioned CNN."""

    def __init__(self, channels: int, layers: int, steps: int):
        super().__init__()
        self.error_input = nn.Conv1d(1, channels, 1)
        self.condition_input = nn.Conv1d(1, channels, 1)
        self.step_embedding = StepEmbedding(steps, channels)
        self.blocks = nn.ModuleList(
            DilatedResidualBlock(channels, 2**i) for i in range(layers)
        )
        self.output = nn.Sequential(
            nn.LeakyReLU(0.2), nn.Conv1d(channels, channels, 1),
            nn.LeakyReLU(0.2), nn.Conv1d(channels, 1, 1)
        )

    def forward(
        self, noisy_error: torch.Tensor, step: torch.Tensor, forecast: torch.Tensor
    ) -> torch.Tensor:
        # Subtraction follows the negative-correlation feature described in Fig. 5.
        hidden = self.error_input(noisy_error.unsqueeze(1))
        hidden = hidden - self.condition_input(forecast.unsqueeze(1))
        hidden = hidden + self.step_embedding(step)
        skips = []
        for block in self.blocks:
            hidden, skip = block(hidden)
            skips.append(skip)
        return self.output(torch.stack(skips).sum(0)).squeeze(1)


class CLDM(nn.Module):
    def __init__(self, config: CLDMConfig):
        super().__init__()
        self.config = config
        self.embedding = EmbeddingNetwork(
            config.nwp_features, config.channels, config.embedding_layers
        )
        self.denoiser = DenoisingNetwork(
            config.channels, config.denoising_layers, config.diffusion_steps
        )
        betas = torch.linspace(config.beta_start, config.beta_end, config.diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_previous = torch.cat([torch.ones(1), alpha_bars[:-1]])
        posterior_variance = betas * (1.0 - alpha_bars_previous) / (1.0 - alpha_bars)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-20))

    def forecast(self, nwp: torch.Tensor) -> torch.Tensor:
        return self.embedding(nwp)

    def diffusion_loss(self, nwp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            forecast = self.forecast(nwp)
        clean_error = target - forecast
        step = torch.randint(0, self.config.diffusion_steps, (len(target),), device=target.device)
        noise = torch.randn_like(clean_error)
        alpha_bar = self.alpha_bars[step].unsqueeze(1)
        noisy = alpha_bar.sqrt() * clean_error + (1.0 - alpha_bar).sqrt() * noise
        predicted_noise = self.denoiser(noisy, step, forecast)
        return nn.functional.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, nwp: torch.Tensor, scenarios: int) -> torch.Tensor:
        forecast = self.forecast(nwp)
        batch, horizon = forecast.shape
        forecast = forecast[:, None, :].expand(batch, scenarios, horizon).reshape(-1, horizon)
        error = torch.randn_like(forecast)
        for index in reversed(range(self.config.diffusion_steps)):
            step = torch.full((len(error),), index, device=error.device, dtype=torch.long)
            predicted_noise = self.denoiser(error, step, forecast)
            alpha = self.alphas[index]
            alpha_bar = self.alpha_bars[index]
            mean = (error - (1.0 - alpha) / (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha.sqrt()
            if index:
                mean = mean + self.posterior_variance[index].sqrt() * torch.randn_like(error)
            error = mean
        generated = (forecast + error).clamp(0.0, 1.0)
        return generated.reshape(batch, scenarios, horizon)

    def checkpoint(self) -> dict:
        return {"config": asdict(self.config), "state_dict": self.state_dict()}
