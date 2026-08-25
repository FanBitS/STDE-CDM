"""Paper-style visualization aligned with Zhou's ``plot_paper`` routine."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("default")
plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "serif"],
    "font.size": 19,
    "legend.fontsize": 18,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "grid.color": "#C9C9C9",
    "grid.linewidth": 1.0,
    "grid.alpha": 0.3,
    "lines.linewidth": 3.0,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#CCCCCC",
    "legend.fancybox": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

TITLE_KW = dict(fontsize=24, fontweight="bold", fontstyle="italic")
LABEL_KW = dict(fontsize=20, fontweight="bold", fontstyle="italic")
LEGEND_KW = dict(prop={
    "family": "STIXGeneral", "size": 18, "weight": "bold", "style": "italic"
})


def plot_wind_paper(
    *,
    gen_power_all: np.ndarray,
    gen_alpha_all: np.ndarray,
    gen_cap_individual: np.ndarray,
    gen_pmin_individual: np.ndarray,
    gen_ramp_rate: np.ndarray,
    gen_cost: np.ndarray,
    wind_pred: np.ndarray,
    wind_error_scenarios_test: np.ndarray,
    method: str,
    epsilon: float,
    theta: float,
    network_name: str,
    output_stem: Path,
) -> dict[str, object]:
    """Create the wind counterpart of the upstream ``plot_paper`` figure.

    The generator selection, test-scenario selection, step conventions, AGC
    bars, and ramp panels intentionally follow the upstream implementation.
    A wind forecast/actual row replaces its optional solar row.
    """
    gen_power_all = np.asarray(gen_power_all)
    gen_alpha_all = np.asarray(gen_alpha_all)
    wind_pred = np.asarray(wind_pred)
    wind_error_scenarios_test = np.asarray(wind_error_scenarios_test)
    horizon, num_gen = gen_power_all.shape

    if horizon < 2:
        raise ValueError("paper-style ramp visualization requires T >= 2")
    if num_gen < 2:
        raise ValueError("paper-style visualization requires at least 2 generators")

    alpha_std = np.std(gen_alpha_all, axis=0)
    plot_gen_index = np.argsort(alpha_std)[-2:][::-1]
    scenario_index = int(
        np.random.RandomState(10).choice(wind_error_scenarios_test.shape[0], 1)[0]
    )
    wind_error = wind_error_scenarios_test.sum(axis=-1)[scenario_index]
    wind_forecast = wind_pred.sum(axis=-1)
    wind_actual = wind_forecast + wind_error

    num_rows = 1 + len(plot_gen_index) * 3
    fig, axs = plt.subplots(num_rows, 1, figsize=(10, 2.52 * num_rows), squeeze=False)
    x = np.arange(horizon)

    ax_w = axs[0, 0]
    ax_w.step(x, wind_forecast, label="forecast", where="post", color="#1F77B4")
    ax_w.step(x, wind_actual, label="actual", where="post", color="#FF7F0E")
    ax_w.set_title("Wind Farm", **TITLE_KW)
    ax_w.set_ylabel("Wind (MW)", **LABEL_KW)
    ax_w.set_xlim(-0.5, horizon - 0.5)
    ax_w.set_xticks(np.arange(horizon))
    ax_w.legend(**LEGEND_KW)
    ax_w.grid(True, linestyle="--", alpha=0.3, linewidth=1.0)

    for ig, g in enumerate(plot_gen_index):
        base_row = 1 + ig * 3
        actual_power = gen_power_all[:, g] - gen_alpha_all[:, g] * wind_error

        ax = axs[base_row, 0]
        ax.step(x, gen_power_all[:, g], label="first-stage", where="post", color="#1F77B4")
        ax.step(x, actual_power, label="actual", where="post", color="#FF7F0E")
        ax.set_title(f"Gen {g}, Cost {gen_cost[g]:.2f} USD/MWh", **TITLE_KW)
        ax.set_ylabel("Gen (MW)", **LABEL_KW)
        ax.axhline(gen_pmin_individual[g], color="black", linestyle="--", linewidth=2.5)
        ax.axhline(gen_cap_individual[g], color="black", linestyle="--", linewidth=2.5)
        ax.set_xlim(-0.5, horizon - 0.5)
        ax.set_xticks(np.arange(horizon))
        ax.legend(**LEGEND_KW)
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=1.0)

        ax = axs[base_row + 1, 0]
        ax.bar(x + 0.5, gen_alpha_all[:, g], width=0.8, label="AGC", color="violet")
        ax.set_title(f"Gen {g} AGC Factor", **TITLE_KW)
        ax.set_ylabel("AGC Factor", **LABEL_KW)
        ax.set_ylim(-1, 1)
        ax.set_xlim(-0.5, horizon - 0.5)
        ax.set_xticks(np.arange(horizon))
        ax.grid(True, linestyle="--", alpha=0.5, linewidth=1.0)
        ax.legend(**LEGEND_KW)

        first_stage_ramp = np.diff(gen_power_all[:, g])
        actual_ramp = np.diff(actual_power)
        ramp_x = np.arange(1, horizon) + 0.5
        width = 0.35
        ramp_limit = gen_ramp_rate[g]

        ax = axs[base_row + 2, 0]
        ax.bar(
            ramp_x - width / 2,
            first_stage_ramp,
            width,
            label="First-stage",
            color="steelblue",
            alpha=0.7,
        )
        ax.bar(
            ramp_x + width / 2,
            actual_ramp,
            width,
            label="Actual",
            color="darkorange",
            alpha=0.7,
        )
        ax.axhline(
            ramp_limit,
            color="red",
            linestyle="--",
            linewidth=2.5,
            label=f"Limit (±{ramp_limit:.1f} MW)",
        )
        ax.axhline(-ramp_limit, color="red", linestyle="--", linewidth=2.5)
        ax.set_ylim(-ramp_limit * 1.2, ramp_limit * 1.2)
        ax.set_title(f"Gen {g} Ramping", **TITLE_KW)
        ax.set_ylabel("Ramp (MW/h)", **LABEL_KW)
        ax.set_xlim(-0.5, horizon - 0.5)
        ax.set_xticks(np.arange(horizon))
        ax.grid(True, linestyle="--", alpha=0.5, linewidth=1.0)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.85), **LEGEND_KW)

    for ax in axs.flat:
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="major", labelsize=19, width=1.0)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

    axs[-1, 0].set_xlabel("Hour", **LABEL_KW)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.97, bottom=0.03, hspace=0.35)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "pdf": str(pdf_path),
        "png": str(png_path),
        "scenario_index": scenario_index,
        "selected_generators": plot_gen_index.tolist(),
        "wind_error_mean": float(wind_error.mean()),
        "wind_error_std": float(wind_error.std()),
    }
