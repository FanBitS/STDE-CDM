# -*- coding: utf-8 -*-
import os
import sys
import csv
import re
import time
import threading
import argparse
import traceback
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

import numpy as np


def _get_rss_bytes():
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except Exception:
        pass
    try:
        with open("/proc/self/statm") as f:
            rss_pages = int(f.read().split()[1])
        return rss_pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0


class PeakMemSampler:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, _get_rss_bytes())
            self._stop.wait(self.interval)

    def __enter__(self):
        self.peak = _get_rss_bytes()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)


DEFAULT_EPS_THETA = [(0.08, 0.12), (0.05, 0.10), (0.10, 0.15), (0.03, 0.06)]
DEFAULT_N = [50, 80, 100, 150, 200, 250, 300]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_METHODS = ["FICA", "EIFICA"]


def parse_args():
    p = argparse.ArgumentParser(description="Parameter-grid evaluation")
    p.add_argument("--eps-theta", default=None,
                   help="semicolon-separated eps,theta pairs, e.g. '0.03,0.06;0.05,0.10'; default: original 4 pairs")
    p.add_argument("--N", default=None, help="comma-separated N_WDR list; default 50..300")
    p.add_argument("--seeds", default=None, help="comma-separated seeds; default 0,1,2,3,4")
    p.add_argument("--methods", default=None, help="comma-separated methods; default FICA,EIFICA")
    p.add_argument("--network", default="case118")
    p.add_argument("--num_gen", type=int, default=38)
    p.add_argument("--num_solar", type=int, default=5)
    p.add_argument("--num_wt", type=int, default=0)
    p.add_argument("--T", type=int, default=24)
    p.add_argument("--norm_ord", type=int, default=1)
    p.add_argument("--load_scaling", type=int, default=1)
    p.add_argument("--jobs", type=int, default=3, help="joblib parallel jobs; default 3 (set 1 if low memory)")
    p.add_argument("--out", default=None,
                   help="output directory; default Solar_results_fixed[_oneshot]")
    p.add_argument("--summarize-only", default=None,
                   help="summarize only: comma-separated result dirs, merge into one CSV then exit")
    return p.parse_args()


def build_grid(args):
    if args.eps_theta:
        eps_theta = []
        for pair in args.eps_theta.split(";"):
            e, t = pair.split(",")
            eps_theta.append((float(e), float(t)))
    else:
        eps_theta = DEFAULT_EPS_THETA
    N_list = [int(x) for x in args.N.split(",")] if args.N else DEFAULT_N
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else DEFAULT_SEEDS
    methods = [m.strip() for m in args.methods.split(",")] if args.methods else DEFAULT_METHODS
    return eps_theta, N_list, seeds, methods


def result_name(args, theta, epsilon, seed, N, method):
    return (f"result_{args.network}_theta{theta}_epsilon{epsilon}_gurobi_seed{seed}"
            f"_num_gen{args.num_gen}_N_WDR{N}_load_scaling_factor{args.load_scaling}"
            f"_{method}_T{args.T}_num_Solar{args.num_solar}.npy")


def solve_one(args, out_dir, theta, epsilon, seed, N, method, tag):
    npy_path = os.path.join(out_dir, result_name(args, theta, epsilon, seed, N, method))
    if os.path.exists(npy_path):
        print(f"{tag} [SKIP] exists: {os.path.basename(npy_path)}", flush=True)
        return
    print(f"{tag} [START] {method} eps={epsilon} theta={theta} seed={seed} N={N}", flush=True)
    from solar_all_method import solve_PD_instance
    peak_rss_mb = float("nan")
    try:
        with PeakMemSampler() as mem:
            res = solve_PD_instance(
                num_gen=args.num_gen, num_WT=args.num_wt, num_Solar=args.num_solar,
                T=args.T, method=method, N_WDR=N, epsilon=epsilon, theta=theta,
                norm_ord=args.norm_ord, load_scaling_factor=args.load_scaling,
                show_plot=False, network_name=args.network, seed=seed,
            )
        peak_rss_mb = mem.peak / (1024.0 * 1024.0)
        rec = {
            "min_cost (USD)": float(res["obj_value"]),
            "reliability_test (%)": float(res.get("satisfied_rate", float("nan"))) * 100.0,
            "t_solve (s)": float(res.get("solve_time", float("nan"))),
        }
        it_log = res.get("iteration_log") or []
        rec["eifica_iters"] = len(it_log) if method == "EIFICA" else ""
        print(f"{tag} [DONE] {method} seed={seed} N={N}: "
              f"cost={rec['min_cost (USD)']:.2f}, JCC={rec['reliability_test (%)']:.1f}%, "
              f"time={rec['t_solve (s)']:.1f}s, peakRSS={peak_rss_mb:.0f}MB", flush=True)
    except Exception as e:
        try:
            peak_rss_mb = mem.peak / (1024.0 * 1024.0)
        except Exception:
            pass
        msg = f"{type(e).__name__}: {e}"
        status = "OOM" if "out of memory" in str(e).lower() else "ERROR"
        print(f"{tag} [{status}] {method} seed={seed} N={N}: {msg}", flush=True)
        traceback.print_exc()
        rec = {"min_cost (USD)": np.nan, "reliability_test (%)": np.nan,
               "t_solve (s)": np.nan, "error": msg}
    rec["peak_rss_mb"] = peak_rss_mb
    rec.update({
        "network": args.network, "epsilon": epsilon, "theta": theta,
        "seed": seed, "N_WDR": N, "method": method, "num_gen": args.num_gen,
        "num_Solar": args.num_solar, "T": args.T,
        "eifica_mode": "one-shot" if os.environ.get("EIFICA_MAX_ITER", "") == "0" else "iterative",
    })
    np.save(npy_path, rec, allow_pickle=True)


_FN_RE = re.compile(
    r"result_(?P<network>.+?)_theta(?P<theta>[\d.]+)_epsilon(?P<epsilon>[\d.]+)"
    r"_gurobi_seed(?P<seed>\d+)_num_gen(?P<num_gen>\d+)_N_WDR(?P<N_WDR>\d+)"
    r"_load_scaling_factor(?P<lsf>[\d.]+)_(?P<method>[A-Za-z]+)_T(?P<T>\d+)"
    r"_num_Solar(?P<num_Solar>\d+)\.npy$"
)

SUMMARY_FIELDS = [
    "network", "method", "eifica_mode", "epsilon", "theta", "seed", "N_WDR",
    "num_gen", "num_Solar", "T", "min_cost", "reliability_pct", "t_solve_s",
    "peak_rss_mb", "eifica_iters", "error", "source_dir", "file",
]


def _row_from_npy(out_dir, fn):
    d = np.load(os.path.join(out_dir, fn), allow_pickle=True).item()
    m = _FN_RE.match(fn)
    g = m.groupdict() if m else {}
    def pick(key, fn_key=None, cast=lambda x: x):
        if key in d and d[key] != "":
            return d[key]
        v = g.get(fn_key or key)
        return cast(v) if v is not None else ""
    return {
        "network": pick("network"),
        "method": pick("method"),
        "eifica_mode": d.get("eifica_mode", ""),
        "epsilon": pick("epsilon", cast=float),
        "theta": pick("theta", cast=float),
        "seed": pick("seed", cast=int),
        "N_WDR": pick("N_WDR", cast=int),
        "num_gen": pick("num_gen", cast=int),
        "num_Solar": pick("num_Solar", cast=int),
        "T": pick("T", cast=int),
        "min_cost": d.get("min_cost (USD)"),
        "reliability_pct": d.get("reliability_test (%)"),
        "t_solve_s": d.get("t_solve (s)"),
        "peak_rss_mb": d.get("peak_rss_mb", ""),
        "eifica_iters": d.get("eifica_iters", ""),
        "error": d.get("error", ""),
        "source_dir": os.path.basename(out_dir.rstrip("/\\")),
        "file": fn,
    }


def summarize(out_dirs, csv_path):
    if isinstance(out_dirs, str):
        out_dirs = [out_dirs]
    rows = []
    for out_dir in out_dirs:
        if not os.path.isdir(out_dir):
            print(f"[summary] skipping non-existent dir: {out_dir}", flush=True)
            continue
        for fn in sorted(os.listdir(out_dir)):
            if fn.startswith("result_") and fn.endswith(".npy"):
                rows.append(_row_from_npy(out_dir, fn))
    if rows:
        rows.sort(key=lambda r: (str(r["method"]), r["epsilon"] if r["epsilon"] != "" else 0,
                                 r["N_WDR"] if r["N_WDR"] != "" else 0,
                                 r["seed"] if r["seed"] != "" else 0))
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"[summary] {len(rows)} instances -> {csv_path}", flush=True)
    else:
        print("[summary] no result_*.npy found", flush=True)


def main():
    args = parse_args()

    if args.summarize_only:
        dirs = [os.path.join(HERE, d.strip()) if not os.path.isabs(d.strip()) else d.strip()
                for d in args.summarize_only.split(",") if d.strip()]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summarize(dirs, os.path.join(HERE, f"master_summary_{stamp}.csv"))
        return

    eps_theta, N_list, seeds, methods = build_grid(args)

    oneshot = os.environ.get("EIFICA_MAX_ITER", "") == "0"
    out_dir = args.out or (f"Solar_results_fixed{'_oneshot' if oneshot else ''}")
    out_dir = os.path.join(HERE, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    tasks = []
    for (epsilon, theta) in eps_theta:
        for N in N_list:
            for seed in seeds:
                for method in methods:
                    tasks.append((theta, epsilon, seed, N, method))
    total = len(tasks)
    mode = "direct (EIFICA_MAX_ITER=0)" if oneshot else "iterative (default)"
    print("=" * 60, flush=True)
    print(f"paper_eval | network={args.network} | mode={mode}", flush=True)
    print(f"eps_theta={eps_theta}", flush=True)
    print(f"N={N_list} seeds={seeds} methods={methods}", flush=True)
    print(f"total instances={total} | out={out_dir} | jobs={args.jobs}", flush=True)
    print("=" * 60, flush=True)

    def tag(i):
        return f"[{i+1}/{total}]"

    if args.jobs and args.jobs > 1:
        from joblib import Parallel, delayed
        Parallel(n_jobs=args.jobs)(
            delayed(solve_one)(args, out_dir, theta, epsilon, seed, N, method, tag(i))
            for i, (theta, epsilon, seed, N, method) in enumerate(tasks)
        )
    else:
        for i, (theta, epsilon, seed, N, method) in enumerate(tasks):
            solve_one(args, out_dir, theta, epsilon, seed, N, method, tag(i))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summarize(out_dir, os.path.join(out_dir, f"summary_{stamp}.csv"))
    print("All done.", flush=True)


if __name__ == "__main__":
    main()
