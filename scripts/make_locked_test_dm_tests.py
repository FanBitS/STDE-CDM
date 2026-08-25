from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "results"
sys.path.insert(0, str(ROOT / "src"))

from stde_cdm import (  # noqa: E402
    JointCLDM,
    JointDDPM,
    JointVAE,
    JointWGANGenerator,
    STJCDM,
    build_joint_umnn,
    load_joint,
)

MODELS = [
    "Joint-WGAN-GP",
    "Joint-VAE",
    "Joint-UMNN",
    "Joint-DDPM",
    "Joint-CLDM",
    "STDE-CDM",
]
METRICS = ["CRPS", "PS", "ES", "VS"]
DISPLAY_MODELS = ["WGAN", "VAE", "UMNN", "DDPM", "CLDM", "STDE"]


def inverse_scaled(generated: np.ndarray, checkpoint: dict, count: int) -> np.ndarray:
    restored = (
        generated.reshape(50, count, -1) * checkpoint["y_std"][None, None, :]
        + checkpoint["y_mean"][None, None, :]
    )
    return np.clip(restored.reshape(50, count, 24, 5), 0, 1)


def scaled_condition(values: np.ndarray, checkpoint: dict) -> torch.Tensor:
    normalized = (
        values.reshape(len(values), -1) - checkpoint["x_mean"]
    ) / checkpoint["x_std"]
    return torch.from_numpy(normalized.astype(np.float32))


def daily_scores(
    scenarios: np.ndarray, observations: np.ndarray, device: torch.device
) -> dict[str, np.ndarray]:
    """Return one score per TEST day for flattened 5-site, 24-hour paths."""
    samples = torch.from_numpy(
        scenarios.reshape(len(observations), scenarios.shape[1], -1)
    ).to(device)
    observed = torch.from_numpy(observations.reshape(len(observations), -1)).to(device)
    count = samples.shape[1]

    absolute = (samples - observed[:, None, :]).abs().mean(dim=1)
    ordered = samples.sort(dim=1).values
    coefficients = (
        2 * torch.arange(1, count + 1, device=device) - count - 1
    ).reshape(1, count, 1)
    pair_term = (ordered * coefficients).sum(dim=1) / (count * count)
    crps = (absolute - pair_term).mean(dim=1)

    quantiles = torch.arange(0.05, 1.0, 0.05, device=device)
    predicted = torch.quantile(samples, quantiles, dim=1)
    residual = observed[None, :, :] - predicted
    pinball = torch.maximum(
        quantiles[:, None, None] * residual,
        (quantiles[:, None, None] - 1.0) * residual,
    ).mean(dim=(0, 2))

    first = torch.linalg.vector_norm(
        samples - observed[:, None, :], dim=2
    ).mean(dim=1)
    second = torch.linalg.vector_norm(
        samples[:, :, None, :] - samples[:, None, :, :], dim=3
    ).mean(dim=(1, 2)) / 2.0
    energy = first - second
    observed_difference = (
        observed[:, :, None] - observed[:, None, :]
    ).abs().sqrt()
    generated_difference = (
        samples[:, :, :, None] - samples[:, :, None, :]
    ).abs().sqrt().mean(dim=1)
    variogram = (
        (observed_difference - generated_difference) ** 2
    ).sum(dim=(1, 2))
    return {
        "CRPS": crps.cpu().numpy(),
        "PS": pinball.cpu().numpy(),
        "ES": energy.cpu().numpy(),
        "VS": variogram.cpu().numpy(),
    }


@torch.no_grad()
def generate_seed(seed: int, count: int, device: torch.device, data) -> dict[str, np.ndarray]:
    raw = torch.from_numpy(data.x_test).to(device)
    models: dict[str, np.ndarray] = {}

    path = ROOT / "artifacts" / "checkpoints" / f"joint_vae_seed{seed}.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = JointVAE(
        latent_size=checkpoint["config"]["latent_size"],
        hidden_size=checkpoint["config"]["hidden_size"],
        hidden_layers=checkpoint["config"]["hidden_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    generator = torch.Generator(device=device).manual_seed(81000 + seed)
    sample = model.sample(
        scaled_condition(data.x_test, checkpoint).to(device), count, generator
    ).cpu().numpy()
    models["Joint-VAE"] = inverse_scaled(sample, checkpoint, count)

    path = (
        ROOT / "artifacts" / "checkpoints" / f"joint_wgan_gp_seed{seed}.pt"
    )
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = JointWGANGenerator(
        latent_size=checkpoint["config"]["latent_size"],
        width=checkpoint["config"]["width"],
        layers=checkpoint["config"]["layers"],
    ).to(device)
    model.load_state_dict(checkpoint["generator_state_dict"])
    model.eval()
    generator = torch.Generator(device=device).manual_seed(82000 + seed)
    sample = model.sample(
        scaled_condition(data.x_test, checkpoint).to(device), count, generator
    ).cpu().numpy()
    models["Joint-WGAN-GP"] = inverse_scaled(sample, checkpoint, count)

    path = ROOT / "artifacts" / "checkpoints" / f"joint_ddpm_seed{seed}.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = {
        key: value for key, value in checkpoint["config"].items()
        if key not in {"epochs", "batch_size", "seed"}
    }
    model = JointDDPM(**config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    generator = torch.Generator(device=device).manual_seed(83000 + seed)
    models["Joint-DDPM"] = model.sample(raw, count, generator).cpu().numpy()

    path = ROOT / "artifacts" / "checkpoints" / f"joint_umnn_seed{seed}.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_joint_umnn().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    condition = scaled_condition(data.x_test, checkpoint).to(device)
    generated = []
    torch.manual_seed(84000 + seed)
    for day in range(50):
        latent = torch.randn(count, 120, device=device)
        generated.append(
            model.invert(latent, condition[day:day + 1].expand(count, -1))
            .cpu().numpy()
        )
    models["Joint-UMNN"] = inverse_scaled(
        np.stack(generated), checkpoint, count
    )

    path = ROOT / "artifacts" / "checkpoints" / f"joint_cldm_seed{seed}.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    distribution = JointCLDM().to(device)
    distribution.load_state_dict(checkpoint["state_dict"])
    distribution.eval()
    torch.manual_seed(85000 + seed)
    distribution_samples = distribution.sample(raw, count).cpu().numpy()
    models["Joint-CLDM"] = distribution_samples

    path = (
        ROOT
        / "artifacts"
        / "checkpoints"
        / f"stde_spatiotemporal_seed{seed}.pt"
    )
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    spatiotemporal = STJCDM(**checkpoint["config"]).to(device)
    spatiotemporal.load_state_dict(checkpoint["state_dict"])
    spatiotemporal.eval()
    generator = torch.Generator(device=device).manual_seed(85000 + seed)
    spatiotemporal_samples = spatiotemporal.sample(
        raw, count, generator
    ).cpu().numpy()
    models["STDE-CDM"] = (
        0.6 * distribution_samples + 0.4 * spatiotemporal_samples
    )
    return models


def dm_matrices(losses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided non-overlapping one-day DM test, averaged over training seeds."""
    # losses: seed, model, metric, day
    daily = losses.mean(axis=0)
    model_count, metric_count, day_count = daily.shape
    pvalues = np.ones((metric_count, model_count, model_count))
    directions = np.zeros_like(pvalues)
    for metric in range(metric_count):
        for row in range(model_count):
            for column in range(model_count):
                if row == column:
                    continue
                differential = daily[row, metric] - daily[column, metric]
                statistic = differential.mean() / (
                    differential.std(ddof=1) / np.sqrt(day_count)
                )
                pvalues[metric, row, column] = 2 * stats.t.sf(
                    abs(statistic), day_count - 1
                )
                directions[metric, row, column] = np.sign(differential.mean())
    return pvalues, directions


def plot(pvalues: np.ndarray, directions: np.ndarray) -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "mathtext.fontset": "stix",
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "serif"],
            "font.size": 19,
            "axes.titlesize": 24,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "grid.color": "#C9C9C9",
            "grid.linestyle": "--",
            "grid.linewidth": 1.0,
            "grid.alpha": 0.3,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # Match the effective source width of Fig. 7.  With one inset colour bar
    # attached to every panel, this makes each heatmap-plus-colour-bar unit
    # occupy the same horizontal span as one Fig. 7 subplot after inclusion.
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 10.5))
    image = None
    for index, (ax, metric) in enumerate(zip(axes.flat, METRICS)):
        # The reference paper plots directional one-sided DM p-values as
        # percentages and caps the colour scale at 10%.  Recover those values
        # from the stored two-sided tests and the loss-difference direction.
        one_sided = np.where(
            directions[index] < 0,
            pvalues[index] / 2.0,
            1.0 - pvalues[index] / 2.0,
        )
        displayed = np.minimum(100.0 * one_sided, 10.0)
        np.fill_diagonal(displayed, np.nan)
        colour_map = plt.get_cmap("RdYlGn_r").copy()
        colour_map.set_bad("#ECECF0")
        image = ax.imshow(displayed, vmin=0, vmax=10, cmap=colour_map, aspect="equal")
        colorbar_axis = inset_axes(
            ax,
            width="4%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(1.04, 0.0, 1.0, 1.0),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        panel_colorbar = fig.colorbar(image, cax=colorbar_axis)
        panel_colorbar.set_ticks([0, 2, 4, 6, 8, 10])
        panel_colorbar.ax.tick_params(labelsize=15, width=0.8, pad=0.5)
        plt.setp(panel_colorbar.ax.get_yticklabels(), fontweight="bold")
        ax.set_xticks(
            np.arange(len(DISPLAY_MODELS)), DISPLAY_MODELS, rotation=48, ha="right"
        )
        ax.set_yticks(np.arange(len(DISPLAY_MODELS)), DISPLAY_MODELS)
        ax.tick_params(length=0)
        plt.setp(ax.get_xticklabels(), fontweight="bold")
        plt.setp(ax.get_yticklabels(), fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(
            0.5, -0.32, f"({chr(97 + index)}) {metric} DM test.",
            transform=ax.transAxes, ha="center", va="top", fontsize=24,
            fontweight="bold", fontstyle="italic",
        )
    fig.subplots_adjust(
        left=0.06, right=0.91, bottom=0.11, top=0.98, wspace=0.52, hspace=0.38
    )
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "locked_test_dm_significance"
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot frozen DM tests or recompute them from all checkpoints"
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Regenerate daily losses from ten model checkpoints before plotting.",
    )
    args = parser.parse_args()
    frozen = OUT / "locked_test_daily_scores_10seed.npz"
    if frozen.is_file() and not args.recompute:
        with np.load(frozen, allow_pickle=False) as payload:
            pvalues = np.asarray(payload["pvalues"], dtype=float)
            directions = np.asarray(payload["directions"], dtype=float)
        plot(pvalues, directions)
        print(f"plotted frozen results from {frozen}", flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_joint(
        ROOT / "data" / "wind_data_all_zone.csv"
    )
    count = 200
    seed_losses = []
    for seed in range(10):
        generated = generate_seed(seed, count, device, data)
        model_losses = []
        for name in MODELS:
            scores = daily_scores(generated[name], data.y_test, device)
            model_losses.append(np.stack([scores[metric] for metric in METRICS]))
        seed_losses.append(np.stack(model_losses))
        print(f"completed seed {seed}", flush=True)
    losses = np.stack(seed_losses)
    pvalues, directions = dm_matrices(losses)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        frozen,
        losses=losses,
        models=np.asarray(MODELS),
        metrics=np.asarray(METRICS),
        pvalues=pvalues,
        directions=directions,
    )
    report = {
        "protocol": (
            "50 locked TEST days; 200 joint scenarios/day; daily losses averaged "
            "over 10 training seeds before pairwise two-sided non-overlapping-day "
            "Diebold-Mariano tests"
        ),
        "models": MODELS,
        "metrics": METRICS,
        "pvalues": pvalues.tolist(),
        "direction": (
            "negative means the row model has lower mean daily loss than the "
            "column model"
        ),
        "mean_daily_losses": losses.mean(axis=(0, 3)).tolist(),
    }
    report_path = OUT / "locked_test_dm_significance.json"
    report_path.write_text(
        json.dumps(report, indent=2)
    )
    plot(pvalues, directions)
    print(f"saved results to {OUT}", flush=True)


if __name__ == "__main__":
    main()
