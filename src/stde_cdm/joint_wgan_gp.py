"""Joint five-farm extension of the source project's conditional WGAN-GP."""
from __future__ import annotations

import torch
from torch import nn


def _linear_stack(sizes: list[int], final_relu: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (left, right) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(left, right))
        if index < len(sizes) - 2:
            layers.append(nn.ReLU())
        elif final_relu:
            # Retained from the original Discriminator_wassertein.
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class JointWGANGenerator(nn.Module):
    def __init__(self, condition_size=480, target_size=120, latent_size=64,
                 width=256, layers=2):
        super().__init__()
        self.condition_size = condition_size
        self.target_size = target_size
        self.latent_size = latent_size
        self.network = _linear_stack(
            [latent_size + condition_size, *([width] * layers), target_size]
        )

    def initialize(self, mean=0.0, std=0.02):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=mean, std=std)

    def forward(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        condition = condition.reshape(len(condition), self.condition_size)
        return self.network(torch.cat([noise, condition], dim=1))

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, scenarios: int,
               generator: torch.Generator | None = None) -> torch.Tensor:
        days = len(condition)
        repeated = condition[:, None].expand(
            days, scenarios, *condition.shape[1:]
        ).reshape(days * scenarios, -1)
        noise = torch.randn(days * scenarios, self.latent_size,
                            device=condition.device, dtype=condition.dtype,
                            generator=generator)
        return self(noise, repeated).reshape(days, scenarios, 24, 5)


class JointWGANCritic(nn.Module):
    def __init__(self, condition_size=480, target_size=120, width=256,
                 layers=2, lambda_gp=10.0):
        super().__init__()
        self.condition_size = condition_size
        self.target_size = target_size
        self.lambda_gp = lambda_gp
        self.network = _linear_stack(
            [target_size + condition_size, *([width] * layers), 1],
            final_relu=True,
        )

    def initialize(self, mean=0.0, std=0.02):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=mean, std=std)

    def forward(self, sample: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        sample = sample.reshape(len(sample), self.target_size)
        condition = condition.reshape(len(condition), self.condition_size)
        return self.network(torch.cat([sample, condition], dim=1))

    def gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor,
                         condition: torch.Tensor) -> torch.Tensor:
        real, fake = real.reshape(len(real), -1), fake.reshape(len(fake), -1)
        # The source implementation samples one interpolation weight per value.
        epsilon = torch.rand_like(real)
        mixed = (real * epsilon + fake * (1 - epsilon)).requires_grad_(True)
        score = self(mixed, condition)
        gradient = torch.autograd.grad(
            score, mixed, torch.ones_like(score), create_graph=True,
            retain_graph=True,
        )[0]
        return ((gradient.flatten(1).norm(2, dim=1) - 1) ** 2).mean()

    def loss(self, fake: torch.Tensor, real: torch.Tensor,
             condition: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        real_score = self(real, condition).mean()
        fake_score = self(fake, condition).mean()
        penalty = self.gradient_penalty(real, fake, condition)
        wasserstein = real_score - fake_score
        total = -wasserstein + self.lambda_gp * penalty
        return total, {"wasserstein": wasserstein, "gradient_penalty": penalty}
