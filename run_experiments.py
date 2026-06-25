#!/usr/bin/env python
"""
run_experiments.py — Grid evaluation runner for the DP streaming DFG miner.

Discovers all .xes / .xes.gz files in ``./data/input/``, derives a parameter
grid for each log, and runs ``test_framework.run_evaluation()`` with every
configuration — **in parallel** across CPU cores.

For each event log the expensive oracle DFG and event data are pre-computed
**once** before spawning the pool, and passed to every worker.

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
    python run_experiments.py --workers 4              # limit parallelism
"""

import argparse
import glob
import json
import multiprocessing
import os
import re
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pm4py


def _parse_interval(interval_str: str) -> timedelta:
    """Parse a human-friendly interval string into a :class:`timedelta`.

    Same logic as ``test_framework.parse_publish_interval`` but inlined
    to avoid pulling in heavy transitive imports (cvxpy, pm4py streaming, …).
    """
    m = re.fullmatch(r'(\d+(?:\.\d+)?)\s*([smhd])', interval_str.strip().lower())
    if not m:
        raise ValueError(
            f"Invalid publish interval '{interval_str}'. "
            "Use e.g. '30s', '5m', '2h', or '1d'."
        )
    value = float(m.group(1))
    unit = m.group(2)
    multipliers = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
    return timedelta(seconds=value * multipliers[unit])

# ── Parameter grid knobs ─────────────────────────────────────────────────────

WINDOW_FRACTIONS   = [0.01, 0.05, 0.10]        # W as fraction of N
RATIOS             = [1, 2, 5]                  # r = W / P
TRACE_QUANTILES    = [                          # (label, quantile)
    ("Q75",  0.75),
    ("Q90",  0.90),
    ("Q100", 1.00),
]

# ── Time-based grid knobs ────────────────────────────────────────────────────
TIME_PUB_COUNTS      = [20, 50, 100]               # data-derived: ~N publications over the log
TIME_FIXED_INTERVALS = ["10d", "5d", "1d"]           # fixed calendar intervals
TIME_MIN_PUBS        = 5                             # skip combos producing fewer publications
TIME_MAX_PUBS        = 200                           # skip combos producing more publications

INPUT_DIR  = os.path.join("data", "input")
OUTPUT_DIR = os.path.join("data", "output")


# ── Log analysis ─────────────────────────────────────────────────────────────

def analyze_log(filepath: str) -> dict:
    """Read a XES log and return statistics needed for parameter derivation.

    Returns
    -------
    dict with keys: N, num_cases, trace_quantiles, activities,
                    total_duration_seconds
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

    # Temporal statistics
    first_ts = df["time:timestamp"].min()
    last_ts  = df["time:timestamp"].max()
    total_duration = last_ts - first_ts
    total_duration_seconds = total_duration.total_seconds()

    print(f"    N={N:,}  cases={num_cases:,}  activities={len(activities)}")
    print(f"    duration: {total_duration}  ({total_duration_seconds:.0f}s)")
    for label, val in quantiles.items():
        print(f"    trace length {label} = {val}")

    return {
        "N": N,
        "num_cases": num_cases,
        "trace_quantiles": quantiles,
        "activities": activities,
        "total_duration_seconds": total_duration_seconds,
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


def _seconds_to_interval_str(seconds: float) -> str:
    """Convert a duration in seconds to a compact interval string.

    Prefers whole units (``'2d'``, ``'6h'``) and falls back to
    fractional representations (``'1.5h'``) when necessary.
    """
    if seconds >= 86400 and seconds % 86400 == 0:
        return f"{int(seconds / 86400)}d"
    if seconds >= 3600 and seconds % 3600 == 0:
        return f"{int(seconds / 3600)}h"
    if seconds >= 60 and seconds % 60 == 0:
        return f"{int(seconds / 60)}m"
    # Fractional — pick the most readable unit
    if seconds >= 86400:
        return f"{seconds / 86400:.1f}d"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.0f}s"


def generate_time_configs(stats: dict, epsilon: float) -> list[dict]:
    """Derive time-based (W, interval, L, α) combinations from log statistics.

    Two interval strategies are combined:

    1. **Data-derived**: ``total_duration / N`` for N ∈ TIME_PUB_COUNTS,
       guaranteeing a controlled number of publications regardless of the
       log's temporal extent.
    2. **Fixed calendar**: intervals from TIME_FIXED_INTERVALS, filtered
       to skip combos that would produce fewer than TIME_MIN_PUBS or more
       than TIME_MAX_PUBS estimated publications.

    The budget fraction α is auto-derived: ``α = 1 / r_time`` where
    ``r_time`` is the estimated number of publications per window-worth
    of time, clamped to [0.05, 1.0].
    """
    N = stats["N"]
    quantiles = stats["trace_quantiles"]
    total_seconds = stats["total_duration_seconds"]

    if total_seconds <= 0:
        print("  [WARN] Log has zero temporal extent — skipping time-based configs.")
        return []

    median_event_rate = N / total_seconds          # events / second
    configs: list[dict] = []
    seen_names: set[str] = set()                   # deduplicate

    for w_frac in WINDOW_FRACTIONS:
        W = max(10, int(round(N * w_frac)))
        W_time_seconds = W / median_event_rate     # temporal extent of one window

        # ── Data-derived intervals ────────────────────────────────────
        for n_pubs in TIME_PUB_COUNTS:
            interval_seconds = total_seconds / n_pubs
            interval_str = _seconds_to_interval_str(interval_seconds)

            r_time = W_time_seconds / interval_seconds if interval_seconds > 0 else 1
            alpha_time = 1.0 / max(1, round(r_time))
            alpha_time = max(0.05, min(alpha_time, 1.0))

            for q_label, _q_val in TRACE_QUANTILES:
                L = min(quantiles[q_label], W)
                L = max(L, 2)

                pct_label = f"W{int(w_frac * 100)}pct"
                config_name = f"{pct_label}_T{n_pubs}pubs_L{q_label}"
                if config_name in seen_names:
                    continue
                seen_names.add(config_name)

                configs.append({
                    "name":             config_name,
                    "W":                W,
                    "L":                L,
                    "publish_interval": interval_str,
                    "alpha":            round(alpha_time, 6),
                    "epsilon":          epsilon,
                    "w_frac":           w_frac,
                    "q_label":          q_label,
                    "estimated_pubs":   n_pubs,
                    "interval_seconds": round(interval_seconds, 2),
                })

        # ── Fixed calendar intervals ──────────────────────────────────
        for interval_str in TIME_FIXED_INTERVALS:
            interval_td = _parse_interval(interval_str)
            interval_seconds = interval_td.total_seconds()

            estimated_pubs = total_seconds / interval_seconds
            if estimated_pubs < TIME_MIN_PUBS or estimated_pubs > TIME_MAX_PUBS:
                continue

            r_time = W_time_seconds / interval_seconds if interval_seconds > 0 else 1
            alpha_time = 1.0 / max(1, round(r_time))
            alpha_time = max(0.05, min(alpha_time, 1.0))

            for q_label, _q_val in TRACE_QUANTILES:
                L = min(quantiles[q_label], W)
                L = max(L, 2)

                pct_label = f"W{int(w_frac * 100)}pct"
                config_name = f"{pct_label}_T{interval_str}_L{q_label}"
                if config_name in seen_names:
                    continue
                seen_names.add(config_name)

                configs.append({
                    "name":             config_name,
                    "W":                W,
                    "L":                L,
                    "publish_interval": interval_str,
                    "alpha":            round(alpha_time, 6),
                    "epsilon":          epsilon,
                    "w_frac":           w_frac,
                    "q_label":          q_label,
                    "estimated_pubs":   int(round(estimated_pubs)),
                    "interval_seconds": round(interval_seconds, 2),
                })

    return configs


# ── Worker function (runs in child process) ──────────────────────────────────

# Module-level globals populated before forking — each worker inherits these
# via fork (copy-on-write) rather than pickle-serialising large event lists.
_worker_event_list:   list        = []
_worker_activity_set: set[str]    = set()
_worker_num_cases:    int         = 0
_worker_oracle:       tuple       = ({}, {}, {})


def _init_worker(event_list, activity_set, num_cases, oracle):
    """Pool initializer: store shared read-only data in module globals."""
    global _worker_event_list, _worker_activity_set, _worker_num_cases, _worker_oracle
    _worker_event_list   = event_list
    _worker_activity_set = activity_set
    _worker_num_cases    = num_cases
    _worker_oracle       = oracle


def _run_single_config(task: dict) -> dict:
    """Execute one parameter configuration inside a pool worker.

    *task* is a plain dict so it can be pickled easily for dispatch.

    Returns a status dict with 'name', 'output_path', 'ok', 'elapsed',
    and 'error' (if any).
    """
    # Import here so each forked child has its own module-level state
    from test_framework import run_evaluation

    config       = task["config"]
    output_path  = task["output_path"]
    log_path     = task["log_path"]
    log_file     = task["log_file"]
    max_pubs     = task["max_publications"]
    pub_interval = task.get("publish_interval")
    time_bf      = task.get("time_budget_fraction", 0.4)
    cfg_label    = config["name"]

    result = {"name": cfg_label, "output_path": output_path, "ok": False, "elapsed": 0.0, "error": None}

    t0 = time.monotonic()
    try:
        run_evaluation(
            log_path=log_path,
            log_file=log_file,
            window_size=config["W"],
            epsilon=config["epsilon"],
            max_trace_events=config["L"],
            output_path=output_path,
            publish_period=config.get("P"),
            max_publications=max_pubs,
            budget_fraction=config["alpha"],
            publish_interval=pub_interval,
            time_budget_fraction=time_bf,
            # ── pre-computed data (avoids re-reading the XES + oracle per worker) ──
            precomputed_event_list=_worker_event_list,
            precomputed_activity_set=_worker_activity_set,
            precomputed_num_cases=_worker_num_cases,
            precomputed_oracle=_worker_oracle,
        )
        result["ok"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"    ✗ {cfg_label} FAILED: {result['error']}", file=sys.stderr)

    result["elapsed"] = time.monotonic() - t0
    return result


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
    parser.add_argument(
        "--workers", type=int, default=None,
        help=(
            "Number of parallel worker processes (default: number of CPU cores).  "
            "Set to 1 for sequential execution."
        ),
    )
    parser.add_argument(
        "--publish-interval", type=str, default=None,
        help=(
            "Time-based publishing interval using simulated event timestamps.  "
            "Format: <number><unit> where unit is s/m/h/d.  "
            "Examples: '30s', '5m', '1d'.  Overrides the event-count publish period."
        ),
    )
    parser.add_argument(
        "--time-budget-fraction", type=float, default=0.4,
        help=(
            "Fraction of the total budget spent per publication when using "
            "time-based publishing (default: 0.4)."
        ),
    )
    args = parser.parse_args()

    n_workers = args.workers if args.workers is not None else multiprocessing.cpu_count()
    # Ensure at least 1 worker
    n_workers = max(1, n_workers)

    # ── Discover log files ────────────────────────────────────────────────
    log_files = sorted(
        glob.glob(os.path.join(args.input_dir, "*.xes"))
        + glob.glob(os.path.join(args.input_dir, "*.xes.gz"))
    )

    if not log_files:
        print(f"No .xes or .xes.gz files found in {args.input_dir}/", file=sys.stderr)
        print("Place event logs there and re-run.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(log_files)} log(s) in {args.input_dir}/")
    print(f"Using {n_workers} worker process(es)\n")

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
        time_configs = generate_time_configs(stats, epsilon=args.epsilon)
        out_dir = os.path.join(args.output_dir, log_stem)
        out_dir_time = os.path.join(out_dir, "time_based")
        print(f"\n  {len(configs)} event-count + {len(time_configs)} time-based configurations\n")

        if args.dry_run:
            print("  Event-count configs:")
            for c in configs:
                print(f"    {c['name']:25s}  W={c['W']:>6,}  P={c['P']:>6,}  "
                      f"L={c['L']:>4}  α={c['alpha']:.4f}")
            print("\n  Time-based configs:")
            for c in time_configs:
                print(f"    {c['name']:30s}  W={c['W']:>6,}  "
                      f"interval={c['publish_interval']:>6s}  "
                      f"L={c['L']:>4}  α={c['alpha']:.4f}  "
                      f"~{c['estimated_pubs']} pubs")
            print()
            total_configs += len(configs) + len(time_configs)
            continue

        # ── Create output directories & save metadata ─────────────────
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(out_dir_time, exist_ok=True)

        meta = {
            "log_file": log_name,
            "log_stem": log_stem,
            "epsilon": args.epsilon,
            "log_stats": {
                "N": stats["N"],
                "num_cases": stats["num_cases"],
                "trace_quantiles": stats["trace_quantiles"],
                "num_activities": len(stats["activities"]),
                "total_duration_seconds": stats["total_duration_seconds"],
            },
            "configs": configs,
            "time_configs": time_configs,
        }
        meta_path = os.path.join(out_dir, "_experiment_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"  Metadata → {meta_path}")

        # ── Pre-compute shared data ONCE for this log ─────────────────
        # This is the key optimisation: the XES file is read and the
        # oracle DFG is mined in the main process; workers inherit it.
        from test_framework import load_log_data, compute_offline_dfg

        print("  Pre-computing shared data for all configs …")
        t_pre = time.monotonic()
        event_list, activity_set, num_cases = load_log_data(log_dir, log_name)
        oracle_dfg, oracle_sa, oracle_ea = compute_offline_dfg(log_dir, log_name, activity_set)
        oracle = (oracle_dfg, oracle_sa, oracle_ea)
        print(f"  Pre-computation done in {time.monotonic() - t_pre:.1f}s\n")

        # ── Build task list (skip existing outputs) ───────────────────
        tasks = []
        skipped = 0

        # Event-count configs
        for cfg in configs:
            output_path = os.path.join(out_dir, f"{cfg['name']}.json")

            if os.path.exists(output_path):
                print(f"  {cfg['name']}  — SKIP (exists)")
                skipped += 1
                continue

            tasks.append({
                "config":              cfg,
                "output_path":         output_path,
                "log_path":            log_dir,
                "log_file":            log_name,
                "max_publications":    args.max_publications,
                "publish_interval":    args.publish_interval,
                "time_budget_fraction": args.time_budget_fraction,
            })

        # Time-based configs
        for cfg in time_configs:
            output_path = os.path.join(out_dir_time, f"{cfg['name']}.json")

            if os.path.exists(output_path):
                print(f"  {cfg['name']}  — SKIP (exists)")
                skipped += 1
                continue

            tasks.append({
                "config":              cfg,
                "output_path":         output_path,
                "log_path":            log_dir,
                "log_file":            log_name,
                "max_publications":    args.max_publications,
                "publish_interval":    cfg["publish_interval"],
                "time_budget_fraction": cfg["alpha"],
            })

        total_ok += skipped
        total_configs += skipped

        all_config_count = len(configs) + len(time_configs)
        if not tasks:
            print("  All configs already completed — nothing to do.\n")
            summary.append((log_stem, all_config_count))
            continue

        print(f"  Dispatching {len(tasks)} config(s) to {min(n_workers, len(tasks))} worker(s) …\n")

        # ── Run in parallel ───────────────────────────────────────────
        # Use 'fork' context so children inherit the pre-computed data
        # via copy-on-write without pickling the large event list.
        ctx = multiprocessing.get_context("fork")
        effective_workers = min(n_workers, len(tasks))

        t_pool = time.monotonic()
        with ctx.Pool(
            processes=effective_workers,
            initializer=_init_worker,
            initargs=(event_list, activity_set, num_cases, oracle),
        ) as pool:
            results = pool.map(_run_single_config, tasks)

        pool_elapsed = time.monotonic() - t_pool

        # ── Summarise results ─────────────────────────────────────────
        ok_count   = sum(1 for r in results if r["ok"])
        fail_count = sum(1 for r in results if not r["ok"])

        total_configs += len(tasks)
        total_ok      += ok_count
        total_fail    += fail_count

        print(f"\n  Log '{log_stem}': {ok_count}/{len(tasks)} succeeded, "
              f"{fail_count} failed  ({pool_elapsed:.1f}s wall-clock)")

        for r in results:
            status = "✓" if r["ok"] else "✗"
            err    = f"  — {r['error']}" if r.get("error") else ""
            print(f"    {status} {r['name']:25s}  {r['elapsed']:.1f}s{err}")

        if fail_count:
            print(f"\n  ⚠ {fail_count} config(s) FAILED for '{log_stem}'", file=sys.stderr)

        summary.append((log_stem, all_config_count))
        print()

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
