#!/usr/bin/env python3
"""Reproducible visualizations from locked results and checkpoints."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
OUT = ROOT / "figures" / "results"

sys.path.insert(0, str(ROOT / "src"))

METRIC_NAMES = ["MAE", "RMSE", "CRPS", "PS", "ES", "VS"]
MODELS = [
    "Joint-WGAN-GP",
    "Joint-VAE",
    "Joint-UMNN",
    "Joint-DDPM",
    "Joint-CLDM",
    "STDE-CDM",
]
COLORS = {
    "Joint-WGAN-GP": "#D62728",
    "Joint-VAE": "#9467BD",
    "Joint-UMNN": "#8C564B",
    "Joint-DDPM": "#2CA02C",
    "Joint-CLDM": "#FF7F0E",
    "STDE-CDM": "#1F77B4",
}

# Figure 4 is intentionally more compact than the other single-column result
# figures. This factor changes only its canvas height; its final typography
# remains normalized to the same visible size as Figures 5--8.
FIG4_HEIGHT_SCALE = 3.0 / 5.0


def configure() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "mathtext.fontset": "stix",
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "serif"],
            "font.size": 19,
            "axes.labelsize": 24,
            "axes.titlesize": 24,
            "legend.fontsize": 18,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.titlepad": 10,
            "axes.labelpad": 6,
            "grid.color": "#C9C9C9",
            "grid.linestyle": "--",
            "grid.linewidth": 1.0,
            "grid.alpha": 0.3,
            "lines.linewidth": 3.0,
            "legend.frameon": True,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # All result PDFs are ultimately scaled to the same IEEE column width.
    # Normalize against the 705.05 pt tightly cropped width of the FICA
    # ``Wind Farm'' reference.  These factors use the actual exported PDF
    # widths rather than figsize, which is essential for Fig. 6 because its
    # inset colour bars and right > 1 layout enlarge the tight bounding box.
    paper_figure_scale = {
        "locked_test_six_metric_comparison": 450.775 / 705.05,      # Fig. 4
        "locked_test_day1_joint_scenario_bands": 692.535 / 705.05, # Fig. 5
        "locked_test_scenario_shape_temporal_correlation":
            893.417 / 705.05,                                     # Fig. 6
        "locked_test_day1_denoising_evolution": 695.563 / 705.05,  # Fig. 7
    }
    typography_scale = paper_figure_scale.get(
        stem, fig.get_figwidth() / (705.05 / 72.0)
    )
    for ax in fig.axes:
        ax.title.set_fontsize(24 * typography_scale)
        ax.title.set_fontweight("bold")
        ax.title.set_fontstyle("italic")
        ax.xaxis.label.set_fontsize(24 * typography_scale)
        ax.xaxis.label.set_fontweight("bold")
        ax.xaxis.label.set_fontstyle("italic")
        ax.yaxis.label.set_fontsize(24 * typography_scale)
        ax.yaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontstyle("italic")
        ax.tick_params(
            axis="both", which="major",
            labelsize=19 * typography_scale, width=1.0,
        )
        plt.setp(ax.get_xticklabels(), fontweight="bold", fontstyle="normal")
        plt.setp(ax.get_yticklabels(), fontweight="bold", fontstyle="normal")
        if not ax.images:
            ax.grid(True, linestyle="--", alpha=0.3, linewidth=1.0)
    for legend in list(fig.legends) + [
        ax.get_legend() for ax in fig.axes if ax.get_legend() is not None
    ]:
        if legend is not None:
            for text_item in legend.get_texts():
                text_item.set_fontfamily("STIXGeneral")
                text_item.set_fontsize(18 * typography_scale)
                text_item.set_fontweight("bold")
                text_item.set_fontstyle("italic")
    pdf_path = OUT / f"{stem}.pdf"
    png_path = OUT / f"{stem}.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def aligned_bottom_legend(
    fig: plt.Figure,
    handles,
    labels,
    *,
    ncol: int,
    gap_inches: float = 0.18,
) -> None:
    """Align a bottom legend to the visible axes union with a fixed gap."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_position() for ax in fig.axes if ax.get_visible()]
    tight_boxes = [
        ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        for ax in fig.axes if ax.get_visible()
    ]
    x0 = min(box.x0 for box in boxes)
    x1 = max(box.x1 for box in boxes)
    # Measure the gap below the complete decorated axis, including ticks and
    # the x-axis label, rather than below the bare plotting rectangle.
    y0 = min(box.y0 for box in tight_boxes)
    gap = gap_inches / fig.get_figheight()
    fig.legend(
        handles,
        labels,
        ncol=ncol,
        mode="expand",
        loc="upper left",
        bbox_to_anchor=(x0, y0 - gap, x1 - x0, 0.0),
        bbox_transform=fig.transFigure,
        borderaxespad=0.0,
    )


def model_name(name: str) -> str:
    return "STDE-CDM" if name in {"ST-MoE-CDM", "STDE-CDM"} else name


def locked_comparison() -> None:
    payload = json.loads((METRICS / "paper_locked_test_10seed.json").read_text())
    rows = payload["rows"]
    values: dict[str, dict[str, list[float]]] = {
        model: {metric: [] for metric in METRIC_NAMES} for model in MODELS
    }
    for row in rows:
        for raw_name, scores in row["models"].items():
            name = model_name(raw_name)
            if name in values:
                for metric in METRIC_NAMES:
                    values[name][metric].append(scores[metric])

    # Compact Figure 4 to three fifths of its former height. Typography is
    # deliberately not reduced because all publication figures must retain the
    # same final visible font hierarchy at IEEE single-column width.
    fig, ax = plt.subplots(figsize=(7.2, 5.45 * FIG4_HEIGHT_SCALE))
    baseline = "Joint-CLDM"
    proposed = "STDE-CDM"
    reductions = []
    lower_errors = []
    upper_errors = []
    rng = np.random.default_rng(20260724)
    for metric in METRIC_NAMES:
        base = np.asarray(values[baseline][metric])
        ours = np.asarray(values[proposed][metric])
        reduction = 100.0 * (base.mean() - ours.mean()) / base.mean()
        reductions.append(reduction)
        # Paired bootstrap interval for the ratio-of-means estimand reported in
        # the text and table.  Resampling indices preserves seed pairing.
        indices = rng.integers(0, len(base), size=(10000, len(base)))
        boot_base = base[indices].mean(axis=1)
        boot_ours = ours[indices].mean(axis=1)
        boot = 100.0 * (boot_base - boot_ours) / boot_base
        low, high = np.percentile(boot, [2.5, 97.5])
        lower_errors.append(reduction - low)
        upper_errors.append(high - reduction)
    y = np.arange(len(METRIC_NAMES))
    ax.errorbar(
        reductions, y, xerr=np.vstack([lower_errors, upper_errors]),
        fmt="o", markersize=7.0,
        color=COLORS["STDE-CDM"], markeredgecolor="black",
        markeredgewidth=0.55, ecolor="#4A4A4A", elinewidth=2.5, capsize=4.0,
    )
    ax.axvline(0, color="#777777", linewidth=2.5, linestyle="--")
    ax.set_yticks(y, METRIC_NAMES)
    ax.invert_yaxis()
    ax.margins(y=0.14)
    ax.set_xlabel("Reduction relative to Joint CLDM (%)")
    ax.set_title("Gain over the strongest external baseline", pad=9)
    ax.grid(axis="x", color="#C9C9C9", linestyle="--", linewidth=1.0, alpha=0.3)
    ax.set_axisbelow(True)
    right = max(np.asarray(reductions) + np.asarray(upper_errors))
    ax.set_xlim(0, max(4.4, right + 0.35))
    for yi, value in zip(y, reductions):
        ax.annotate(
            f"{value:.2f}%",
            xy=(value, yi),
            xytext=(7, 4),
            textcoords="offset points",
            fontsize=15 * (450.775 / 705.05),
            ha="left",
            va="bottom",
            fontweight="bold",
        )

    fig.subplots_adjust(left=0.19, right=0.95, bottom=0.20, top=0.84)
    save(fig, "locked_test_six_metric_comparison")


def expert_complementarity() -> None:
    payload = json.loads((METRICS / "dual_expert_locked_test_weight_0p4.json").read_text())
    rows = payload["rows"]
    keys = ["distribution_expert", "spatiotemporal_expert", "fused"]
    labels = ["Distribution", "Spatiotemporal", "Fused STDE-CDM"]
    colors = ["#8064A2", "#4E9DB3", "#167D9A"]
    means = {
        key: np.array([np.mean([row[key][m] for row in rows]) for m in METRIC_NAMES])
        for key in keys
    }
    reference = means["distribution_expert"]
    normalized = {key: 100 * means[key] / reference for key in keys}

    x = np.arange(len(METRIC_NAMES))
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, color, marker in zip(keys[1:], labels[1:], colors[1:], ["o", "s"]):
        change = normalized[key] - 100
        ax.plot(x, change, marker=marker, markersize=4, linewidth=0.9,
                color=color, label=label)
    ax.axhline(0, color="#555555", linewidth=0.7, linestyle="--")
    ax.set_xticks(x, METRIC_NAMES)
    ax.set_ylabel("Change relative to\ndistribution expert (%)")
    ax.set_ylim(-4.5, 1.5)
    ax.grid(color="#D9D9D9", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, loc="lower left")
    fig.tight_layout()
    save(fig, "locked_test_expert_complementarity")


def scenario_bands() -> None:
    from stde_cdm import (
        JointCLDM,
        JointWGANGenerator,
        STJCDM,
        build_joint_umnn,
        load_joint,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_joint(ROOT / "data" / "wind_data_all_zone.csv")
    nwp = torch.from_numpy(data.x_test[:1]).to(device)
    truth = data.y_test[0]

    base_ckpt = torch.load(
        ROOT / "artifacts" / "checkpoints" / "joint_cldm_seed0.pt",
        map_location=device,
        weights_only=False,
    )
    distribution = JointCLDM().to(device)
    distribution.load_state_dict(base_ckpt["state_dict"])
    distribution.eval()

    wgan_ckpt = torch.load(
        ROOT / "artifacts" / "checkpoints" / "joint_wgan_gp_seed0.pt",
        map_location=device,
        weights_only=False,
    )
    wgan = JointWGANGenerator(
        latent_size=wgan_ckpt["config"]["latent_size"],
        width=wgan_ckpt["config"]["width"],
        layers=wgan_ckpt["config"]["layers"],
    ).to(device)
    wgan.load_state_dict(wgan_ckpt["generator_state_dict"])
    wgan.eval()

    umnn_ckpt = torch.load(
        ROOT / "artifacts" / "checkpoints" / "joint_umnn_seed0.pt",
        map_location=device,
        weights_only=False,
    )
    umnn = build_joint_umnn().to(device)
    umnn.load_state_dict(umnn_ckpt["state_dict"])
    umnn.eval()

    def scaled_day(checkpoint):
        flattened = data.x_test[:1].reshape(1, -1)
        normalized = (flattened - checkpoint["x_mean"]) / checkpoint["x_std"]
        return torch.from_numpy(normalized.astype(np.float32)).to(device)

    def restore_day(generated, checkpoint):
        restored = (
            generated.reshape(200, -1) * checkpoint["y_std"][None, :]
            + checkpoint["y_mean"][None, :]
        )
        return np.clip(restored.reshape(200, 24, 5), 0, 1)

    st_ckpt = torch.load(
        ROOT / "artifacts" / "checkpoints" / "stde_spatiotemporal_seed0.pt",
        map_location=device,
        weights_only=False,
    )
    spatiotemporal = STJCDM(**st_ckpt["config"]).to(device)
    spatiotemporal.load_state_dict(st_ckpt["state_dict"])
    spatiotemporal.eval()

    @torch.no_grad()
    def paired_trace(model, condition_input, scenarios, generator, is_st):
        if is_st:
            forecast, condition = model.encoder(condition_input)
        else:
            forecast, condition = model.forecast(condition_input), None
        _, hours_count, farms_count = forecast.shape
        expanded_forecast = forecast[:, None].expand(
            1, scenarios, hours_count, farms_count
        ).reshape(-1, hours_count, farms_count)
        expanded_condition = None
        if condition is not None:
            expanded_condition = condition[:, None].expand(
                1, scenarios, *condition.shape[1:]
            ).reshape(-1, *condition.shape[1:])
        error = torch.randn(expanded_forecast.shape, device=device, generator=generator)
        trace = {model.steps: (expanded_forecast + error).clamp(0, 1)}
        for index in reversed(range(model.steps)):
            step = torch.full((len(error),), index, device=device, dtype=torch.long)
            if is_st:
                predicted = model.denoiser(
                    error, step, expanded_forecast, expanded_condition
                )
            else:
                predicted = model.denoiser(error, step, expanded_forecast)
            alpha, alpha_bar = model.alphas[index], model.alpha_bars[index]
            error = (
                error - (1 - alpha) / (1 - alpha_bar).sqrt() * predicted
            ) / alpha.sqrt()
            if index:
                error = error + model.posterior_variance[index].sqrt() * torch.randn(
                    error.shape, device=device, generator=generator
                )
            if index in {30, 10}:
                trace[index] = (expanded_forecast + error).clamp(0, 1)
        trace[0] = (expanded_forecast + error).clamp(0, 1)
        samples = trace[0].reshape(1, scenarios, hours_count, farms_count)
        trace = {
            step: value.reshape(1, scenarios, hours_count, farms_count)
            for step, value in trace.items()
        }
        return forecast, samples, trace

    sampling_seed = 85000
    dist_forecast_t, dist_samples_t, dist_trace = paired_trace(
        distribution, nwp, 200,
        torch.Generator(device=device).manual_seed(sampling_seed), False
    )
    st_forecast_t, st_samples_t, st_trace = paired_trace(
        spatiotemporal, nwp, 200,
        torch.Generator(device=device).manual_seed(sampling_seed), True
    )
    dist_samples = dist_samples_t.cpu().numpy()[0]
    st_samples = st_samples_t.cpu().numpy()[0]
    with torch.no_grad():
        wgan_raw = wgan.sample(
            scaled_day(wgan_ckpt),
            200,
            torch.Generator(device=device).manual_seed(82000),
        ).cpu().numpy()
        wgan_samples = restore_day(wgan_raw, wgan_ckpt)

        torch.manual_seed(84000)
        umnn_latent = torch.randn(200, 120, device=device)
        umnn_raw = umnn.invert(
            umnn_latent,
            scaled_day(umnn_ckpt).expand(200, -1),
        ).cpu().numpy()
        umnn_samples = restore_day(umnn_raw, umnn_ckpt)
    fused = 0.6 * dist_samples + 0.4 * st_samples
    dist_forecast = dist_forecast_t.cpu().numpy()[0]
    st_forecast = st_forecast_t.cpu().numpy()[0]
    fused_forecast = 0.6 * dist_forecast + 0.4 * st_forecast

    OUT.mkdir(parents=True, exist_ok=True)
    scenario_data = OUT / "locked_test_day1_seed0_scenarios.npz"
    np.savez_compressed(
        scenario_data,
        truth=truth,
        joint_wgan_gp=wgan_samples,
        joint_umnn=umnn_samples,
        distribution=dist_samples,
        spatiotemporal=st_samples,
        fused=fused,
        fused_forecast=fused_forecast,
        test_day_index=np.array(0),
        training_seed=np.array(0),
        sampling_seed=np.array(sampling_seed),
    )

    hours = np.arange(1, 25)
    fig, axes = plt.subplots(3, 2, figsize=(10, 11), sharex=True, sharey=True)
    axes = axes.flat
    for farm, ax in enumerate(axes):
        if farm == 5:
            aggregate = fused.mean(axis=2)
            for scenario in aggregate[:60]:
                ax.plot(hours, scenario, color="#AFAFAF", linewidth=0.8, alpha=0.34)
            ax.plot(hours, np.median(aggregate, axis=0), color="#2CA02C",
                    linewidth=3.0)
            ax.plot(hours, fused_forecast.mean(axis=1), color="#1F77B4",
                    linewidth=3.0)
            ax.plot(hours, truth.mean(axis=1), color="#FF7F0E", linewidth=3.0)
            ax.set_title("(f) Five-zone mean")
            ax.set_ylabel("Wind power")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(color="#DDDDDD", linewidth=0.35)
            ax.set_xlabel("Forecast hour")
            ax.set_xticks([1, 6, 12, 18, 24])
            continue
        q05, q25, q50, q75, q95 = np.quantile(
            fused[:, :, farm], [0.05, 0.25, 0.50, 0.75, 0.95], axis=0
        )
        for scenario in fused[:60]:
            ax.plot(hours, scenario[:, farm], color="#AFAFAF", linewidth=0.8, alpha=0.34)
        ax.plot(hours, q50, color="#2CA02C", linewidth=3.0, label="Scenario median")
        ax.plot(hours, fused_forecast[:, farm], color="#1F77B4", linewidth=3.0,
                label="Deterministic forecast")
        ax.plot(hours, truth[:, farm], color="#FF7F0E", linewidth=3.0, label="Observation")
        ax.set_title(f"({chr(97 + farm)}) Zone {farm + 1}")
        ax.set_ylabel("Wind power")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(color="#DDDDDD", linewidth=0.35)
        if farm >= 4:
            ax.set_xlabel("Forecast hour")
        ax.set_xticks([1, 6, 12, 18, 24])
    handles, labels = axes[0].get_legend_handles_labels()
    grey = plt.Line2D([0], [0], color="#B8B8B8", linewidth=0.7, label="Generated scenarios")
    fig.tight_layout(rect=(0, 0.135, 1, 1), h_pad=0.30, w_pad=0.35)
    aligned_bottom_legend(
        fig, [grey] + handles, ["Generated scenarios"] + labels, ncol=2
    )
    save(fig, "locked_test_day1_joint_scenario_bands")

    # Scenario-shape and temporal-dependence diagnostic inspired by the
    # normalizing-flow comparison paper.  The five-site mean retains one
    # synchronized 24-hour path per joint scenario and avoids selecting a
    # particularly favourable individual site.
    diagnostic_models = [
        ("Joint WGAN-GP", wgan_samples),
        ("Joint UMNN", umnn_samples),
        ("STDE-CDM", fused),
    ]
    observed_mean = truth.mean(axis=1)
    fig, axes = plt.subplots(
        # Use the same effective source width as Fig. 8.  Under the width
        # ratios below, each square heatmap plus its inset colour bar then
        # matches one Fig. 8 panel, while the trajectory column expands into
        # all remaining horizontal space.
        3, 2, figsize=(10.2, 14.25),
        # The right-hand grid cell is sized first to reproduce the normalized
        # square-panel width used by Fig. 8.  The trajectory column receives
        # all of the remaining width and is intentionally rectangular.
        gridspec_kw={"width_ratios": [1.839, 1.0]},
    )
    for row, (label, samples) in enumerate(diagnostic_models):
        aggregate = samples.mean(axis=2)
        q10, q50, q90 = np.quantile(aggregate, [0.10, 0.50, 0.90], axis=0)
        curve_ax, corr_ax = axes[row]
        for scenario in aggregate[:60]:
            curve_ax.plot(
                hours, scenario, color="#B7B7B7", linewidth=0.8, alpha=0.34
            )
        curve_ax.plot(
            hours, q10, color="#1F77B4", linewidth=2.5,
            label="10% quantile",
        )
        curve_ax.plot(hours, q50, color="#1F77B4", linewidth=3.0, label="Median")
        curve_ax.plot(
            hours, q90, color="#2CA02C", linewidth=2.5,
            label="90% quantile",
        )
        curve_ax.plot(
            hours, observed_mean, color="#FF7F0E", linewidth=3.0,
            label="Observation",
        )
        curve_ax.set_xlim(1, 24)
        curve_ax.set_ylim(-0.02, 1.02)
        curve_ax.set_anchor("E")
        curve_ax.set_xticks([1, 6, 12, 18, 24])
        curve_ax.set_ylabel("Wind power")
        curve_ax.text(
            0.98, 0.94, label, transform=curve_ax.transAxes,
            ha="right", va="top",
            fontsize=15 * (893.417 / 705.05),
            fontfamily="STIXGeneral",
            fontweight="bold",
            fontstyle="italic",
            bbox={
                "boxstyle": "round,pad=0.40",
                "facecolor": "white",
                "edgecolor": "#CCCCCC",
                "linewidth": 0.8,
                "alpha": 0.92,
            },
        )
        curve_ax.grid(color="#DDDDDD", linewidth=0.35)
        if row == 0:
            curve_ax.set_title("(a) Scenario trajectories")
        if row == 2:
            curve_ax.set_xlabel("Forecast hour")

        correlation = np.corrcoef(aggregate, rowvar=False)
        correlation_image = corr_ax.imshow(
            correlation, vmin=-1, vmax=1, cmap="RdYlGn",
            origin="lower", interpolation="nearest", aspect="equal",
        )
        corr_ax.set_box_aspect(1)
        colorbar_axis = inset_axes(
            corr_ax,
            width="4%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(1.04, 0.0, 1.0, 1.0),
            bbox_transform=corr_ax.transAxes,
            borderpad=0,
        )
        row_colorbar = fig.colorbar(correlation_image, cax=colorbar_axis)
        corr_ax.set_anchor("W")
        row_colorbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        row_colorbar.ax.tick_params(labelsize=15, width=0.8, pad=0.5)
        plt.setp(row_colorbar.ax.get_yticklabels(), fontweight="bold")
        corr_ax.set_xticks([0, 5, 11, 17, 23], [1, 6, 12, 18, 24])
        corr_ax.set_yticks([0, 5, 11, 17, 23], [1, 6, 12, 18, 24])
        if row == 0:
            corr_ax.set_title("(b) Temporal correlation")
        if row == 2:
            corr_ax.set_xlabel("Forecast hour")
        corr_ax.set_ylabel("")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    grey = plt.Line2D([0], [0], color="#B7B7B7", linewidth=0.7)
    fig.subplots_adjust(
        left=0.06, right=1.055, bottom=0.15, top=0.98, hspace=0.22, wspace=0.12
    )
    aligned_bottom_legend(
        fig, [grey] + handles, ["Generated scenarios"] + labels, ncol=3
    )
    save(fig, "locked_test_scenario_shape_temporal_correlation")

    trace_steps = [50, 30, 10, 0]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.1), sharex=True, sharey=True)
    for panel_index, (ax, step) in enumerate(zip(axes.flat, trace_steps)):
        fused_state = (
            0.6 * dist_trace[step].cpu().numpy()[0]
            + 0.4 * st_trace[step].cpu().numpy()[0]
        )
        for scenario in fused_state[:30]:
            ax.plot(hours, scenario[:, 0], color="#B8B8B8", linewidth=0.8, alpha=0.6)
        ax.plot(hours, truth[:, 0], color="#FF7F0E", linewidth=3.0, label="Observation")
        ax.plot(hours, fused_forecast[:, 0], color="#1F77B4", linewidth=3.0,
                label="Deterministic forecast")
        ax.set_title(f"({chr(97 + trace_steps.index(step))}) Reverse state $n={step}$")
        ax.set_xlim(1, 24)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([1, 6, 12, 18, 24])
        if panel_index >= 2:
            ax.set_xlabel("Forecast hour")
        ax.set_ylabel("Wind power")
        ax.grid(color="#DDDDDD", linewidth=0.35)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    grey = plt.Line2D([0], [0], color="#B8B8B8", linewidth=0.7, label="Scenario states")
    fig.subplots_adjust(
        left=0.13, right=0.985, bottom=0.20, top=0.94,
        wspace=0.24, hspace=0.30,
    )
    aligned_bottom_legend(
        fig, [grey] + handles, ["Scenario states"] + labels, ncol=3
    )
    save(fig, "locked_test_day1_denoising_evolution")


def main() -> None:
    configure()
    locked_comparison()
    expert_complementarity()
    scenario_bands()
    print(f"Saved reproducible figures and data to {OUT}")


if __name__ == "__main__":
    main()
