"""Run a GEFCom wind scenario case through the copied FICA dispatch model."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandapower as pp
import pandapower.networks as ppnw
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.makePTDF import makePTDF

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPTIMIZER_DIR = PROJECT_ROOT / "fica_dispatch_optimizer"
if str(OPTIMIZER_DIR) not in sys.path:
    sys.path.insert(0, str(OPTIMIZER_DIR))

from solar_all_method import check_JCC, solve_PD  # noqa: E402
from .fica_data import prepare_wind_case
from .fica_visualization import plot_wind_paper


NETWORKS = {
    "case5": ppnw.case5,
    "case24_ieee_rts": ppnw.case24_ieee_rts,
    "case118": ppnw.case118,
}

DEFAULT_SCENARIO = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "wind_UMNN_M_1_z0-1-2-3-4_d0_n6000.npz"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wind scenario -> FICA dispatch")
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--train-pool-size", type=int, default=1000)
    parser.add_argument("--test-pool-size", type=int, default=5000)
    parser.add_argument("--network", choices=NETWORKS, default="case24_ieee_rts")
    parser.add_argument("--method", choices=["FICA", "EIFICA", "CVAR"], default="FICA")
    parser.add_argument("--zone", type=int, default=0, help="zero-based wind zone")
    parser.add_argument("--day", type=int, default=0, help="zero-based test day")
    parser.add_argument("--T", type=int, default=24)
    parser.add_argument("--t-start", type=int, default=0)
    parser.add_argument("--num-gen", type=int, default=38)
    parser.add_argument("--n-wdr", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--theta", type=float, default=0.06)
    parser.add_argument("--wind-share", type=float, default=0.45)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=14400.0)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def build_system(args: argparse.Namespace) -> dict:
    if args.t_start < 0 or args.t_start + args.T > 24:
        raise ValueError("t-start and T must define a window within 24 hours")
    if not 0 < args.wind_share < 1:
        raise ValueError("wind-share must be strictly between 0 and 1")

    net = NETWORKS[args.network]()
    pp.rundcpp(net)
    _, ppci = _pd2ppc(net)
    bus_info = ppci["bus"]
    branch_info = ppci["branch"]
    ptdf = makePTDF(
        ppci["baseMVA"], bus_info, branch_info, using_sparse_solver=False
    )

    load_curve_48 = np.load(OPTIMIZER_DIR / "data" / "UK_norm_load_curve_highest.npy")
    load_curve_24 = np.mean(
        np.vstack([load_curve_48[::2], load_curve_48[1::2]]), axis=0
    )
    load_bus_size = bus_info[:, 2].astype(float)
    load_total = float(np.sum(load_bus_size))
    time_slice = slice(args.t_start, args.t_start + args.T)
    load_bus_all = load_curve_24[time_slice, None] * load_bus_size[None, :]

    fixed_rng = np.random.RandomState(0)
    cost_rng = np.random.RandomState(0)
    gen_capacity = fixed_rng.uniform(0.6, 1.4, args.num_gen) * (
        load_total / args.num_gen
    )
    gen_pmin = 0.1 * gen_capacity
    gen_ramp = 0.6 * gen_capacity
    buses = np.arange(bus_info.shape[0])
    gen_buses = fixed_rng.choice(buses, args.num_gen, replace=True)
    line_limit = np.clip(np.abs(branch_info[:, 5]), 0, 2 * load_total)

    wind_capacity = args.wind_share * load_total
    if args.scenario.suffix.lower() == ".npz":
        pool = np.load(args.scenario)
        scenarios = np.asarray(pool["scenarios_pu"], dtype=float)
        if scenarios.ndim != 3 or scenarios.shape[1] != 24:
            raise ValueError(
                "NPZ scenarios_pu must have shape (scenario, 24, wind_farm)"
            )
        required = args.train_pool_size + args.test_pool_size
        if scenarios.shape[0] < required:
            raise ValueError(
                f"scenario pool has {scenarios.shape[0]} trajectories; {required} required"
            )
        train_scenarios = scenarios[: args.train_pool_size]
        test_scenarios = scenarios[
            args.train_pool_size : args.train_pool_size + args.test_pool_size
        ]
        point = np.median(train_scenarios, axis=0)
        time_slice = slice(args.t_start, args.t_start + args.T)
        per_farm_capacity = wind_capacity / scenarios.shape[2]
        wind = {
            "WT_pred": point[time_slice] * per_farm_capacity,
            "WT_error_scenarios_train": (
                train_scenarios[:, time_slice] - point[None, time_slice]
            ) * per_farm_capacity,
            "WT_error_scenarios_test": (
                test_scenarios[:, time_slice] - point[None, time_slice]
            ) * per_farm_capacity,
        }
        num_wt = scenarios.shape[2]
    else:
        bundle = prepare_wind_case(
            args.scenario,
            zone=args.zone,
            day=args.day,
            train_fraction=args.train_fraction,
            seed=args.seed,
        )
        wind = bundle.to_mw(
            wind_capacity, horizon=args.T, start_hour=args.t_start
        )
        num_wt = 1
    wind_buses = fixed_rng.choice(buses, num_wt, replace=True)
    if args.n_wdr > wind["WT_error_scenarios_train"].shape[0]:
        raise ValueError(
            f"n-wdr={args.n_wdr} exceeds available training scenarios "
            f"({wind['WT_error_scenarios_train'].shape[0]})"
        )

    return {
        "T": args.T,
        "num_gen": args.num_gen,
        "num_WT": num_wt,
        "num_branch": len(branch_info),
        "load_bus_all": load_bus_all,
        "PTDF": ptdf,
        "gen_cap_individual": gen_capacity,
        "gen_pmin_individual": gen_pmin,
        "WT_pred": wind["WT_pred"],
        "WT_error_scenarios_train": wind["WT_error_scenarios_train"],
        "WT_error_scenarios_test": wind["WT_error_scenarios_test"],
        "P_line_limit": line_limit,
        "gen_bus_list": gen_buses,
        "WT_bus_list": wind_buses,
        "N_WDR": args.n_wdr,
        "epsilon": args.epsilon,
        "theta": args.theta,
        "MIPGap": 0.001,
        "rng": np.random.RandomState(args.seed),
        "bigM": 1e5,
        "gen_cost": cost_rng.uniform(23.13, 57.03, args.num_gen),
        "gen_cost_quadra": cost_rng.uniform(0.002, 0.008, args.num_gen),
        "gurobi_seed": 0,
        "method": args.method,
        "thread": 4,
        "norm_ord": 1,
        "num_Solar": 0,
        "Solar_pred": None,
        "Solar_error_scenarios_train": None,
        "Solar_bus_list": None,
        "gen_ramp_rate": gen_ramp,
        "time_limit": args.time_limit,
        "wind_capacity_mw": wind_capacity,
    }


def main() -> None:
    args = build_parser().parse_args()
    zone_label = str(args.zone)
    scenario_day = args.day
    if args.scenario.suffix.lower() == ".npz":
        with np.load(args.scenario) as pool_meta:
            if "zones" in pool_meta:
                zone_label = "-".join(map(str, np.asarray(pool_meta["zones"]).tolist()))
            if "day" in pool_meta:
                scenario_day = int(np.asarray(pool_meta["day"]).item())
    system = build_system(args)
    test_errors = system.pop("WT_error_scenarios_test")
    wind_capacity = system.pop("wind_capacity_mw")

    result = solve_PD(**system)
    prob = result["prob"]
    if prob.Status not in {
        gp.GRB.OPTIMAL,
        gp.GRB.TIME_LIMIT,
        gp.GRB.SUBOPTIMAL,
    } or prob.SolCount == 0:
        raise RuntimeError(f"Gurobi ended without a feasible solution: status={prob.Status}")

    generation_raw = result["gen_power_all"]
    alpha_raw = result["gen_alpha_all"]
    generation = np.asarray(
        generation_raw.X if hasattr(generation_raw, "X") else generation_raw
    )
    alpha = np.asarray(alpha_raw.X if hasattr(alpha_raw, "X") else alpha_raw)
    reliability = check_JCC(
        system["T"],
        system["num_gen"],
        system["num_branch"],
        generation,
        alpha,
        system["load_bus_all"],
        system["PTDF"],
        system["gen_cap_individual"],
        system["gen_pmin_individual"],
        system["WT_pred"],
        test_errors,
        system["P_line_limit"],
        system["gen_bus_list"],
        system["WT_bus_list"],
        gen_ramp_rate=system["gen_ramp_rate"],
    )

    args.output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"wind_{args.method.lower()}_{args.network}_z{zone_label}_d{scenario_day}_{stamp}"
    np.savez_compressed(
        args.output / f"{stem}.npz",
        gen_power=generation,
        gen_alpha=alpha,
        wind_pred=system["WT_pred"],
        train_errors=system["WT_error_scenarios_train"],
        test_errors=test_errors,
        load_bus_all=system["load_bus_all"],
        gen_capacity=system["gen_cap_individual"],
        gen_pmin=system["gen_pmin_individual"],
        gen_ramp=system["gen_ramp_rate"],
        gen_cost=system["gen_cost"],
        gen_cost_quadra=system["gen_cost_quadra"],
        gen_bus_list=system["gen_bus_list"],
        wind_bus_list=system["WT_bus_list"],
    )
    figure_info = None
    if not args.no_plot:
        figure_info = plot_wind_paper(
            gen_power_all=generation,
            gen_alpha_all=alpha,
            gen_cap_individual=system["gen_cap_individual"],
            gen_pmin_individual=system["gen_pmin_individual"],
            gen_ramp_rate=system["gen_ramp_rate"],
            gen_cost=system["gen_cost"],
            wind_pred=system["WT_pred"],
            wind_error_scenarios_test=test_errors,
            method=args.method,
            epsilon=args.epsilon,
            theta=args.theta,
            network_name=args.network,
            output_stem=args.output / "figures" / f"{stem}_paper",
        )
    summary = {
        "scenario": str(args.scenario.resolve()),
        "network": args.network,
        "method": args.method,
        "zones": zone_label,
        "day": scenario_day,
        "T": args.T,
        "t_start": args.t_start,
        "num_gen": args.num_gen,
        "n_wdr": args.n_wdr,
        "epsilon": args.epsilon,
        "theta": args.theta,
        "wind_capacity_mw": wind_capacity,
        "objective": float(prob.ObjVal),
        "reliability": float(reliability),
        "solve_time_seconds": float(result.get("solve_time", prob.Runtime)),
        "gurobi_status": int(prob.Status),
        "paper_figure": figure_info,
    }
    (args.output / f"{stem}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved: {args.output / stem}")


if __name__ == "__main__":
    main()
