#!/usr/bin/env python
"""
run_experiments.py — Grid evaluation runner for the DP streaming DFG miner.

Discovers all .xes / .xes.gz files in ``./data/input/``, derives a parameter
grid for each log, and runs ``test_framework.py`` with every configuration.

Parameter grid
--------------
  W  (window size)       :  1 %, 5 %, 10 % of total events N
  r  (W / P ratio)       :  1, 2, 5
  P  (publish period)    :  W / r   (number of events between publications)
  L  (max trace events)  :  75th, 90th, 100th percentile of trace lengths
  α  (budget fraction)   :  1 / r   (ensures sustainable budget under reclaim)

Output layout
-------------
  ./data/output/<log_stem>/W<pct>pct_r<r>_L<q>.json

Usage
-----
    python run_experiments.py                          # run everything
    python run_experiments.py --epsilon 2.0            # custom epsilon
    python run_experiments.py --dry-run                # print configs only
    python run_experiments.py --max-publications 50    # cap per run
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pm4py

# ── Parameter grid knobs ─────────────────────────────────────────────────────

WINDOW_FRACTIONS   = [0.01, 0.05, 0.10]        # W as fraction of N
RATIOS             = [1, 2, 5]                  # r = W / P
TRACE_QUANTILES    = [                          # (label, quantile)
    ("Q75",  0.75),
    ("Q90",  0.90),
    ("Q100", 1.00),
]

INPUT_DIR  = os.path.join("data", "input")
OUTPUT_DIR = os.path.join("data", "output")


# ── Log analysis ─────────────────────────────────────────────────────────────

def analyze_log(filepath: str) -> dict:
    """Read a XES log and return statistics needed for parameter derivation.

    Returns
    -------
    dict with keys: N, num_cases, trace_quantiles, activities
    """
    print(f"  Analyzing {filepath} …", flush=True)
    df = pm4py.read_xes(filepath)
    df.sort_values("time:timestamp", inplace=True, kind="mergesort")

    N = len(df)
    trace_lengths = df.groupby("case:concept:name").size()
    num_cases = len(trace_lengths)

    quantiles = {}
    for label, q in TRACE_QUANTILES:
        quantiles[label] = int(np.ceil(trace_lengths.quantile(q)))

    activities = list(pd.unique(df["concept:name"]))

    print(f"    N={N:,}  cases={num_cases:,}  activities={len(activities)}")
    for label, val in quantiles.items():
        print(f"    trace length {label} = {val}")

    return {
        "N": N,
        "num_cases": num_cases,
        "trace_quantiles": quantiles,
        "activities": activities,
    }


# ── Configuration generation ─────────────────────────────────────────────────

def generate_configs(stats: dict, epsilon: float) -> list[dict]:
    """Derive all (W, r, P, L, α) combinations from log statistics."""
    N = stats["N"]
    quantiles = stats["trace_quantiles"]
    configs = []

    for w_frac in WINDOW_FRACTIONS:
        W = max(10, int(round(N * w_frac)))

        for r in RATIOS:
            P = max(1, W // r)
            alpha = 1.0 / r          # budget fraction for sustainability

            for q_label, _q_val in TRACE_QUANTILES:
                L = quantiles[q_label]
                # L must be ≤ W (hard constraint of the sliding-window miner)
                L = min(L, W)
                L = max(L, 2)         # need at least 2 events per trace

                pct_label = f"W{int(w_frac * 100)}pct"
                config_name = f"{pct_label}_r{r}_L{q_label}"

                configs.append({
                    "name":       config_name,
                    "W":          W,
                    "r":          r,
                    "P":          P,
                    "L":          L,
                    "alpha":      round(alpha, 6),
                    "epsilon":    epsilon,
                    "w_frac":     w_frac,
                    "q_label":    q_label,
                })

    return configs


# ── Execution ─────────────────────────────────────────────────────────────────

def run_single(
    log_path: str,
    log_file: str,
    config: dict,
    output_path: str,
    max_publications: int,
) -> bool:
    """Run test_framework.py for one parameter configuration.

    Returns True on success, False on failure.
    """
    cmd = [
        sys.executable, "test_framework.py",
        "--log-path",          log_path,
        "--log-file",          log_file,
        "--window-size",       str(config["W"]),
        "--epsilon",           str(config["epsilon"]),
        "--max-trace-events",  str(config["L"]),
        "--publish-period",    str(config["P"]),
        "--budget-fraction",   str(config["alpha"]),
        "--max-publications",  str(max_publications),
        "--output",            output_path,
    ]

    print(f"    ▸ {' '.join(cmd)}", flush=True)
    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.monotonic() - t0

    if result.returncode == 0:
        print(f"    ✓ done in {elapsed:.1f}s → {output_path}")
        return True
    else:
        print(f"    ✗ FAILED (exit {result.returncode}) after {elapsed:.1f}s",
              file=sys.stderr)
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Grid runner: evaluate DP streaming DFG miner across parameter combinations."
    )
    parser.add_argument(
        "--epsilon", type=float, default=1.0,
        help="Total privacy budget ε (default: 1.0).",
    )
    parser.add_argument(
        "--max-publications", type=int, default=100,
        help="Max publications per run (default: 100).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print parameter grid without executing anything.",
    )
    parser.add_argument(
        "--input-dir", default=INPUT_DIR,
        help=f"Directory containing XES logs (default: {INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help=f"Root output directory (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    # ── Discover log files ────────────────────────────────────────────────
    log_files = sorted(
        glob.glob(os.path.join(args.input_dir, "*.xes"))
        + glob.glob(os.path.join(args.input_dir, "*.xes.gz"))
    )

    if not log_files:
        print(f"No .xes or .xes.gz files found in {args.input_dir}/", file=sys.stderr)
        print("Place event logs there and re-run.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(log_files)} log(s) in {args.input_dir}/\n")

    # ── Process each log ──────────────────────────────────────────────────
    total_configs = 0
    total_ok      = 0
    total_fail    = 0
    summary       = []

    for log_filepath in log_files:
        log_dir  = os.path.dirname(log_filepath)
        log_name = os.path.basename(log_filepath)
        log_stem = Path(log_name).stem
        # Handle .xes.gz double extension
        if log_stem.endswith(".xes"):
            log_stem = log_stem[:-4]

        print(f"{'=' * 70}")
        print(f"Log: {log_name}  (stem: {log_stem})")
        print(f"{'=' * 70}")

        stats   = analyze_log(log_filepath)
        configs = generate_configs(stats, epsilon=args.epsilon)
        out_dir = os.path.join(args.output_dir, log_stem)
        os.makedirs(out_dir, exist_ok=True)

        # Save the log statistics & derived configs as metadata
        meta = {
            "log_file": log_name,
            "log_stem": log_stem,
            "epsilon": args.epsilon,
            "log_stats": {
                "N": stats["N"],
                "num_cases": stats["num_cases"],
                "trace_quantiles": stats["trace_quantiles"],
                "num_activities": len(stats["activities"]),
            },
            "configs": configs,
        }
        meta_path = os.path.join(out_dir, "_experiment_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"\n  Metadata → {meta_path}")
        print(f"  {len(configs)} configurations to run\n")

        if args.dry_run:
            for c in configs:
                print(f"    {c['name']:25s}  W={c['W']:>6,}  P={c['P']:>6,}  "
                      f"L={c['L']:>4}  α={c['alpha']:.4f}")
            print()
            total_configs += len(configs)
            continue

        for i, cfg in enumerate(configs, 1):
            output_path = os.path.join(out_dir, f"{cfg['name']}.json")

            # Skip if output already exists (resume-friendly)
            if os.path.exists(output_path):
                print(f"  [{i}/{len(configs)}] {cfg['name']}  — SKIP (exists)")
                total_ok += 1
                total_configs += 1
                continue

            print(f"\n  [{i}/{len(configs)}] {cfg['name']}  "
                  f"W={cfg['W']:,}  P={cfg['P']:,}  L={cfg['L']}  α={cfg['alpha']:.4f}")

            ok = run_single(
                log_path=log_dir,
                log_file=log_name,
                config=cfg,
                output_path=output_path,
                max_publications=args.max_publications,
            )
            total_configs += 1
            if ok:
                total_ok += 1
            else:
                total_fail += 1

        summary.append((log_stem, len(configs)))

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    for stem, n in summary:
        print(f"  {stem}: {n} configurations")
    print(f"\n  Total: {total_configs} configs  |  ✓ {total_ok} ok  |  ✗ {total_fail} failed")

    if args.dry_run:
        print("\n  (dry-run mode — nothing was executed)")


if __name__ == "__main__":
    main()
