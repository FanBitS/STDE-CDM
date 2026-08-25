#!/usr/bin/env python3
"""Regenerate the FICA trace from the frozen formal backtest.

The figure uses the STDE CDM constraint informed policy on the first locked
TEST day. No model is trained and no optimization problem is resolved here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "scripts"),
    str(ROOT / "src"),
    str(ROOT / "fica_dispatch_optimizer"),
]

from run_fica_real_backtest import base_fica_system  # noqa: E402


CASE = (
    ROOT
    / "results"
    / "fica"
    / "day_00"
    / "stde_cdm_witness_200.npz"
)
OUTPUT = ROOT / "figures" / "results" / "fig_fica_dispatch_case"

COLORS = {
    "forecast": "#1F77B4",
    "observation": "#FF7F0E",
    "first_stage": "steelblue",
    "actual_dispatch": "darkorange",
    "agc": "violet",
    "limit": "red",
}

TITLE_KW = dict(fontsize=24, fontweight="bold", fontstyle="italic")
LABEL_KW = dict(fontsize=24, fontweight="bold", fontstyle="italic")
LEGEND_KW = dict(
    prop={
        "family": "STIXGeneral",
        "size": 18,
        "weight": "bold",
        "style": "italic",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the paper FICA trace from the real TEST day."
    )
    parser.add_argument(
        "--generator-selection",
        choices=("alpha_std", "alpha_range", "mean_abs_agc"),
        default="alpha_std",
        help=(
            "alpha_std preserves the original figure; alpha_range selects "
            "the largest high to low variation in AGC participation factors; "
            "mean_abs_agc selects the largest mean absolute realized AGC "
            "adjustment in MW."
        ),
    )
    parser.add_argument(
        "--agc-axis",
        choices=("fixed", "tight", "balanced"),
        default="fixed",
        help=(
            "fixed preserves the original [-1, 1] AGC axis; tight scales each "
            "panel closely; balanced reveals hourly variation with additional "
            "vertical headroom."
        ),
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix for a non-destructive figure variant.",
    )
    parser.add_argument(
        "--agc-color",
        default=COLORS["agc"],
        help="Matplotlib color used for the AGC bars.",
    )
    parser.add_argument(
        "--agc-edge-color",
        default="none",
        help="Optional edge color used for the AGC bars.",
    )
    return parser.parse_args()


def configure() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "mathtext.fontset": "stix",
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "serif"],
            "font.size": 19,
            "axes.titlesize": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "legend.fontsize": 18,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "grid.color": "#C9C9C9",
            "grid.linewidth": 1.0,
            "lines.linewidth": 3.0,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#CCCCCC",
            "legend.fancybox": True,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes, grid_alpha: float) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", linewidth=1.0, alpha=grid_alpha)
    ax.tick_params(axis="both", which="major", labelsize=19, width=1.0)
    plt.setp(
        ax.get_xticklabels() + ax.get_yticklabels(),
        fontweight="bold",
        fontstyle="normal",
    )


def main() -> None:
    args = parse_args()
    configure()
    if not CASE.is_file():
        raise FileNotFoundError(CASE)

    with np.load(CASE, allow_pickle=False) as result:
        forecast_pu = np.asarray(result["forecast_pu"], dtype=float)
        observation_pu = np.asarray(result["observation_pu"], dtype=float)
        scheduled = np.asarray(result["scheduled_generation_mw"], dtype=float)
        actual = np.asarray(result["real_actual_generation_mw"], dtype=float)
        alpha = np.asarray(result["alpha"], dtype=float)
        actual_ramp_all = np.asarray(result["real_actual_ramp_mw"], dtype=float)

    settings = SimpleNamespace(
        scenarios=200,
        epsilon=0.03,
        theta=0.06,
        wind_share=0.45,
        time_limit=14400.0,
    )
    system = base_fica_system(settings)
    total_wind_capacity = float(system["wind_capacity_mw"])
    farm_capacity = total_wind_capacity / forecast_pu.shape[1]
    forecast = forecast_pu.sum(axis=1) * farm_capacity
    observation = observation_pu.sum(axis=1) * farm_capacity

    realized_agc_adjustment = actual - scheduled
    if args.generator_selection == "mean_abs_agc":
        selection_score = np.mean(np.abs(realized_agc_adjustment), axis=0)
    elif args.generator_selection == "alpha_range":
        selection_score = np.ptp(alpha, axis=0)
    else:
        selection_score = np.std(alpha, axis=0)
    generator_indices = np.argsort(selection_score)[-2:][::-1]

    suffix = args.output_suffix.strip()
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    output = Path(str(OUTPUT) + suffix)

    balanced_agc_upper = None
    if args.agc_axis == "balanced":
        balanced_agc_upper = max(
            0.20,
            1.65 * float(alpha[:, generator_indices].max()),
        )

    hours = np.arange(24)
    ramp_hours = np.arange(1, 24)
    fig, axes = plt.subplots(7, 1, figsize=(10, 17.64), squeeze=False)
    axes = axes[:, 0]

    ax = axes[0]
    ax.step(
        hours,
        forecast,
        where="post",
        color=COLORS["forecast"],
        label="forecast",
    )
    ax.step(
        hours,
        observation,
        where="post",
        color=COLORS["observation"],
        label="observation",
    )
    ax.set_title("Wind Farm", **TITLE_KW)
    ax.set_ylabel("Wind (MW)", **LABEL_KW)
    ax.legend(**LEGEND_KW)
    style_axis(ax, 0.3)

    for position, generator in enumerate(generator_indices):
        base = 1 + 3 * position
        cost = float(system["gen_cost"][generator])
        lower = float(system["gen_pmin_individual"][generator])
        upper = float(system["gen_cap_individual"][generator])
        ramp_limit = float(system["gen_ramp_rate"][generator])

        ax = axes[base]
        ax.step(
            hours,
            scheduled[:, generator],
            where="post",
            color=COLORS["first_stage"],
            label="first stage",
        )
        ax.step(
            hours,
            actual[:, generator],
            where="post",
            color=COLORS["actual_dispatch"],
            label="realized",
        )
        ax.axhline(lower, color="black", linestyle="--", linewidth=2.5)
        ax.axhline(upper, color="black", linestyle="--", linewidth=2.5)
        ax.set_title(
            f"Gen {generator}, Cost {cost:.2f} USD/MWh",
            **TITLE_KW,
        )
        ax.set_ylabel("Gen (MW)", **LABEL_KW)
        ax.legend(**LEGEND_KW)
        style_axis(ax, 0.3)

        ax = axes[base + 1]
        ax.bar(
            hours + 0.5,
            alpha[:, generator],
            width=0.8,
            color=args.agc_color,
            edgecolor=args.agc_edge_color,
            linewidth=0.45 if args.agc_edge_color != "none" else 0.0,
            label="AGC",
        )
        ax.set_title(f"Gen {generator} AGC Factor", **TITLE_KW)
        ax.set_ylabel("AGC Factor", **LABEL_KW)
        if args.agc_axis in {"tight", "balanced"}:
            alpha_values = alpha[:, generator]
            alpha_min = float(alpha_values.min())
            alpha_max = float(alpha_values.max())
            alpha_span = max(alpha_max - alpha_min, 0.01)
            if alpha_min >= 0.0:
                if args.agc_axis == "balanced":
                    ax.set_ylim(0.0, balanced_agc_upper)
                else:
                    ax.set_ylim(0.0, max(0.02, alpha_max + 0.12 * alpha_span))
            else:
                padding = (0.40 if args.agc_axis == "balanced" else 0.12) * alpha_span
                ax.set_ylim(alpha_min - padding, alpha_max + padding)
        else:
            ax.set_ylim(-1, 1)
        ax.legend(**LEGEND_KW)
        style_axis(ax, 0.5)

        scheduled_ramp = np.diff(scheduled[:, generator])
        realized_ramp = actual_ramp_all[:, generator]
        width = 0.35
        ax = axes[base + 2]
        ax.bar(
            ramp_hours - width / 2,
            scheduled_ramp,
            width,
            color=COLORS["first_stage"],
            alpha=0.7,
            label="First stage",
        )
        ax.bar(
            ramp_hours + width / 2,
            realized_ramp,
            width,
            color=COLORS["actual_dispatch"],
            alpha=0.7,
            label="Realized",
        )
        ax.axhline(
            ramp_limit,
            color=COLORS["limit"],
            linestyle="--",
            linewidth=2.5,
            label=f"Limit (±{ramp_limit:.1f} MW)",
        )
        ax.axhline(
            -ramp_limit,
            color=COLORS["limit"],
            linestyle="--",
            linewidth=2.5,
        )
        ax.set_ylim(-1.2 * ramp_limit, 1.2 * ramp_limit)
        ax.set_title(f"Gen {generator} Ramping", **TITLE_KW)
        ax.set_ylabel("Ramp (MW/h)", **LABEL_KW)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.85), **LEGEND_KW)
        style_axis(ax, 0.5)

    for ax in axes:
        ax.set_xlim(-0.5, 23.5)
        ax.set_xticks(hours)
    axes[-1].set_xlabel("Hour", **LABEL_KW)

    fig.subplots_adjust(
        left=0.08,
        right=0.95,
        top=0.97,
        bottom=0.03,
        hspace=0.35,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output_pdf = output.with_suffix(".pdf")
    output_png = output.with_suffix(".png")
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"case={CASE}")
    print(f"generator_selection={args.generator_selection}")
    print(f"agc_axis={args.agc_axis}")
    print(f"agc_color={args.agc_color}")
    print(f"generators={generator_indices.tolist()}")
    print(f"selection_scores={selection_score[generator_indices].tolist()}")
    print(f"pdf={output_pdf}")
    print(f"png={output_png}")


if __name__ == "__main__":
    main()
