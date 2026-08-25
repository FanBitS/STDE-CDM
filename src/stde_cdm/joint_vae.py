"""Conditional VAE baseline for joint five-farm wind scenarios.

This is the 5-farm extension of the project's original ``VAElinear``:
the ELBO, MLP encoder/decoder and Gaussian latent prior are unchanged in
principle, while target and condition vectors cover all farms jointly.
"""
from __future__ import annotations

import torch
from torch import nn


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (left, right) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(left, right))
        if index < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class JointVAE(nn.Module):
    """One latent vector jointly generates a complete 24 x farm trajectory."""

    def __init__(
        self,
        hours: int = 24,
        farms: int = 5,
        features: int = 4,
        latent_size: int = 20,
        hidden_size: int = 200,
        hidden_layers: int = 1,
    ) -> None:
        super().__init__()
        self.hours = hours
        self.farms = farms
        self.features = features
        self.latent_size = latent_size
        self.target_size = hours * farms
        self.condition_size = hours * farms * features
        hidden = [hidden_size] * hidden_layers
        self.encoder = _mlp(
            [self.target_size + self.condition_size, *hidden, 2 * latent_size]
        )
        self.decoder = _mlp(
            [latent_size + self.condition_size, *hidden, self.target_size]
        )

    def _flatten_condition(self, condition: torch.Tensor) -> torch.Tensor:
        return condition.reshape(len(condition), self.condition_size)

    def encode(
        self, target: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(torch.cat(
            [target.reshape(len(target), self.target_size),
             self._flatten_condition(condition)], dim=1
        ))
        return encoded.chunk(2, dim=1)

    def decode(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        output = self.decoder(torch.cat(
            [latent, self._flatten_condition(condition)], dim=1
        ))
        return output.reshape(-1, self.hours, self.farms)

    def loss(
        self, condition: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Original VAElinear ELBO: summed reconstruction MSE plus KL."""
        mean, log_variance = self.encode(target, condition)
        latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        reconstruction = self.decode(latent, condition)
        reconstruction_loss = ((reconstruction - target) ** 2).flatten(1).sum(1).mean()
        kl = -0.5 * (1 + log_variance - mean.square() - log_variance.exp())
        kl_loss = kl.sum(1).mean()
        total = reconstruction_loss + kl_loss
        return total, {"reconstruction": reconstruction_loss, "kl": kl_loss}

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        scenarios: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        days = len(condition)
        repeated = condition[:, None].expand(
            days, scenarios, *condition.shape[1:]
        ).reshape(-1, *condition.shape[1:])
        latent = torch.randn(
            days * scenarios,
            self.latent_size,
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
        return self.decode(latent, repeated).reshape(
            days, scenarios, self.hours, self.farms
        )
