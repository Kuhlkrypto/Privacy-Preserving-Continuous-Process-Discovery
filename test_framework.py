"""
test_framework.py — Evaluation harness for the DP streaming DFG miner.

Streams a real XES log through the PausableLiveEventStream and publishes
the differentially-private DFG using a **budget-gated** schedule: a
publication fires only when

  (a) the miner's remaining privacy budget ≥ epsilon_per_pub, AND
  (b) at least ``publish_period`` events have arrived since the last publication.

This avoids the pathological behaviour of a fixed publish-every-N schedule
where the budget drains after 3 publications (α=0.3) and all subsequent
snapshots are taken with a noise scale of sensitivity / ε_remaining ≈ ∞.

publish_period defaults to window_size so that the reclaim cycle (~w events)
has time to refill the budget before the next snapshot is taken, sustaining
an indefinite publication rate of roughly one snapshot per window.

At each publication point the following are recorded:

  DFG metrics (current window, noisy vs. non-private baselines):
    - MAE             – mean absolute edge-count error
    - MRE             – mean relative error (normalised by true total count)
    - edge_recall     – fraction of true edges present in the noisy DFG
    - edge_precision  – fraction of noisy edges that are true positives
    - f1 / accuracy   – combined edge-structure metrics

  Two non-private baselines are maintained:
    1. **Offline oracle DFG** – mined from the *entire* test log before streaming
       begins using pm4py's standard DFG discovery.  This is a fixed, immutable
       reference that does not change across publications and represents the best
       possible DFG that can ever be obtained from this dataset.
    2. **Re-replay window baseline** – at each publication the *exact* current
       window of events (from WindowLogRecorder) is replayed through a fresh
       binary_count instance.  This gives the non-private equivalent of the
       private windowed miner on identical data — the fairest peer comparison.
    3. **Accumulating streaming baseline** – the existing binary_count that has
       seen all events so far (kept for continuity).

  Flow conservation:
    - flow_imbalance  – |Σ(START→v) − Σ(u→END)| before/after Kirchhoff fix

  Distribution distance:
    - earth_movers_distance – discrete EMD between edge-frequency distributions

  Process-model quality (Inductive Miner and Heuristics Miner):
    Evaluated against BOTH the rolling-window log and the full log:
    - fitness, precision, generalization, simplicity, F1

All results are appended to a list and written to a JSON file at the end.

Usage
-----
    python test_framework.py --log-path /home/fabian/Github/data --log-file "Sepsis Cases - Event Log.xes" --window-size 500 --epsilon 1.0 --max-trace-events 30 --output results.json

    # Optionally override the publishing period (default = window_size):
        --publish-period 300

    # Set budget fraction α (default 0.4; set to 1/r for sustainable budget):
        --budget-fraction 0.5
"""

import argparse
import json
import sys
from collections import defaultdict
from typing import Any

import pm4py

from src.non_private_discovery import binary_count as DefaultDFGDiscoveryCF
from src.non_private_discovery.binary_count import apply_windowed as WindowedBaselineCF

from src import utils_custom_dfg as utils
from src.log_recorder import FullLogRecorder, WindowLogRecorder
from src.private_discovery import PrivateBinaryWindowDFG as PrivateMiner
from src.PausableLiveEventStream import PausableLiveEventStream
from src.metrics import (compute_dfg_metrics, compute_model_quality, compute_flow_imbalance,
                         compute_earth_movers_distance, mean_absolute_error, mean_relative_error)


# ---------------------------------------------------------------------------
# FullLogRecorder – accumulates every event seen into a growing event log
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Offline oracle DFG — pre-computed from the full log before streaming
# ---------------------------------------------------------------------------

def compute_offline_dfg(
    log_path: str,
    log_file: str,
    activities: set[str],
) -> tuple[dict, dict, dict]:
    """Mine the DFG from the *entire* test log using pm4py's standard offline
    discovery.  This is computed **once** before the stream starts and used as
    a fixed oracle reference throughout the evaluation.

    Returns
    -------
    (dfg, start_activities, end_activities)
        All values are plain Python dicts with string-tuple keys / string keys.
    """
    import os
    print("  [Oracle] Mining offline full-log DFG …", end=" ", flush=True)
    log_df = pm4py.read_xes(os.path.join(log_path, log_file))
    log_df.sort_values(by=["time:timestamp"], ascending=True, inplace=True, kind="mergesort")
    event_log = pm4py.convert_to_event_log(log_df)

    dfg_obj, sa, ea = pm4py.discover_dfg(event_log)
    # pm4py returns Counter-like objects; normalise to plain dicts
    dfg = dict(dfg_obj)
    sa  = dict(sa)
    ea  = dict(ea)
    print(f"done  ({len(dfg)} edges, {len(sa)} start acts, {len(ea)} end acts)")
    return dfg, sa, ea


def compute_offline_windowed_dfg(
    event_list: list,
    current_idx: int,
    window_size: int,
    case_id_key: str = "case:concept:name",
    activity_key: str = "concept:name",
    timestamp_key: str = "time:timestamp",
) -> tuple[dict, dict, dict]:
    """Compute the ideal non-private DFG from the exact window slice ending at
    *current_idx* in the pre-loaded (sorted) event list.

    Unlike :class:`WindowedBinaryCount` (which maintains state incrementally
    via streaming), this function has random access to the full event list and
    can materialise the perfect window at any publication point without any
    bloom-filter clipping, trace-limit suppression, or Laplace noise.

    This is the *offline windowed oracle* — the theoretical upper bound for
    any windowed streaming approach.

    Returns
    -------
    (dfg, start_activities, end_activities)
        All plain dicts; binary (case-set) counting, same convention as
        :class:`WindowedBinaryCount`.
    """
    start_idx = max(0, current_idx - window_size)
    window_slice = event_list[start_idx:current_idx]  # events already seen (pre-append)

    case_last: dict[str, tuple[str, str]] = {}
    s_act:     dict[str, int]             = {}
    edges:     dict[tuple[str, str], set] = defaultdict(set)

    for event in window_slice:
        case      = str(event.get(case_id_key,   ""))
        activity  = str(event.get(activity_key,  ""))
        timestamp = str(event.get(timestamp_key, ""))

        if case not in case_last:
            s_act[activity] = s_act.get(activity, 0) + 1
        else:
            last_act, _ = case_last[case]
            if last_act != activity:          # no self-loops
                edges[(last_act, activity)].add(case)

        case_last[case] = (activity, timestamp)

    dfg = {edge: len(cases) for edge, cases in edges.items() if cases}
    ea: dict[str, int] = {}
    for _case, (act, _ts) in case_last.items():
        ea[act] = ea.get(act, 0) + 1

    return dfg, s_act, ea


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    log_path: str,
    log_file: str,
    window_size: int,
    epsilon: float,
    max_trace_events: int,
    output_path: str,
    noise_threshold: float = 0.1,
    publish_period: int | None = None,
    max_publications: int = 0,
    budget_fraction: float = 0.4,
) -> None:
    """
    Parameters
    ----------
    publish_period : int, optional
        Publishing period P — number of events between publications.
        Defaults to *window_size* (one publication per window).  Setting
        P = W/r yields r publications per window.
    max_publications : int
        Stop after this many publications (useful for smoke tests).
    budget_fraction : float
        Fraction α of the current budget spent per publication.
        Set to 1/r (where r = W/P) for sustainable budget management.
    """

    #--------------------------------------------------------
    #                      PREAMBLE
    #--------------------------------------------------------
    print(f"Loading log: {log_path}/{log_file}")
    event_stream, activities, num_cases = utils.get_sample_data(
        data_path=log_path, file_name=log_file
    )
    # Convert to list for both iteration and random-access slicing
    # (needed by compute_offline_windowed_dfg at each publication point)
    event_list   = list(event_stream)
    activity_set = set(activities)
    total_events = len(event_list)
    print(f"  {len(activity_set)} activities, {total_events} events total.")

    print("Pre-computing offline oracle DFG from full log …")
    oracle_dfg, oracle_sa, oracle_ea = compute_offline_dfg(log_path, log_file, activity_set)
    stream = PausableLiveEventStream()
    #--------------------------------------------------------
    #                      MINER AND BASELINE(S) SETUP
    #--------------------------------------------------------
    # --- set up streaming components ---
    miner    = PrivateMiner.apply(
        window_size,
        epsilon,
        max_trace_events=max_trace_events,
        max_cases=num_cases,
        max_error_rate=0.01,
        activities=activity_set,
        budget_fraction=budget_fraction,
    )
    # --- Pre-compute the offline oracle DFG (fixed reference, never changes) ---
    stream.register(miner)

    # Non-private accumulating baseline — same growing scope as the private AggDFG;
    # fair peer comparison that isolates the DP cost of the accumulated result.
    baseline_full_log_cf = DefaultDFGDiscoveryCF.apply(activities=activities)
    stream.register(baseline_full_log_cf)
    # Non-private windowed peer baseline — same window slice as the private miner, no noise, no bloom filter
    windowed_baseline_cf    = WindowedBaselineCF(window_size=window_size, activities=activity_set)
    stream.register(windowed_baseline_cf)


    #--------------------------------------------------------
    #                      LOG RECORDER
    #--------------------------------------------------------
    full_rec = FullLogRecorder()
    stream.register(full_rec)
    win_rec = WindowLogRecorder.apply(window_size=window_size)
    stream.register(win_rec)



    #--------------------------------------------------------
    #                      STREAM
    #--------------------------------------------------------

    # Budget-gated publication schedule.
    # A publication fires when BOTH conditions hold:
    #   1. The remaining budget can cover one more noisy DFG release.
    #   2. At least publish_period events have accumulated since the last publication
    #      (gives the reclaim mechanism time to restore budget for the next cycle).
    if publish_period is None:
        publish_period = window_size   # one publication per window by default




    stream.start()
    results: list[dict[str, Any]] = []
    events_since_publish = 0
    publication_index    = 0

    for j, event in enumerate(event_list):

        budget_ready    = miner.budget > 0
        interval_ready  = events_since_publish >= publish_period
        end_of_stream   = (j == total_events - 1)

        if (budget_ready and interval_ready) or end_of_stream:
            stream.pause()
            print(f"\n[Publication #{publication_index}]  events seen: {j}")

            # ---------- private miner output ----------
            # pm4py's StreamingAlgorithm.get() catches ALL exceptions internally
            # and returns None — guard against that before unpacking.
            # Record budget BEFORE miner.get() so we can compute the exact ε spent here.
            _budget_before = miner.budget
            miner_result = miner.get()
            if miner_result is None:
                print("  [SKIP] miner.get() returned None (budget exhausted or error).",
                      file=sys.stderr)
                events_since_publish = 0
                publication_index += 1
                stream.resume()
                continue

            # NOISED WINDOWED DFG
            unnoised_dfg, raw_noisy_dfg, _dummy_sa, _dummy_ea = miner_result

            # ACCUMULATED / AGGREGATED DFG (flow-corrected)
            clean_acc_dfg, clean_acc_sa, clean_acc_ea = miner.get_current_result()
            filtered_sa = utils.filter_start_activities(clean_acc_sa, noise_threshold=noise_threshold) # FILTERED START ACTIVITIES

            # ACCUMULATED / AGGREGATED DFG (raw — no flow optimisation)
            raw_acc_dfg, raw_acc_sa, raw_acc_ea = miner.get_current_result_no_flow()
            filtered_raw_acc_sa = utils.filter_start_activities(raw_acc_sa, noise_threshold=noise_threshold)

            # FLOW CORRECTED DFG
            # process to dfg with correct flow ( sum(in) = sum(out) )
            noisy_dfg = utils.optimize_flow(raw_noisy_dfg, set(activities))
            noisy_clean, noisy_clean_sa, noisy_clean_ea = utils.remove_dummy_start_and_end_transitions(noisy_dfg)

            # --------------------------------------------------------
            #                      UNNOISED BASELINES
            # --------------------------------------------------------
            # Pass the unnoised DFG through the exact same Kirchhoff optimization.
            # This is the perfect baseline: the exact same algorithm without Laplace noise.
            unnoised_flow_dfg = utils.optimize_flow(unnoised_dfg, set(activities))
            unnoised_clean, unnoised_clean_sa, unnoised_clean_ea = utils.remove_dummy_start_and_end_transitions(unnoised_flow_dfg)
            # Guard: an empty window produces a 0-edge DFG whose 100% precision / 0% recall
            # is meaningless — mark all downstream metrics as None instead.
            _unnoised_empty = not unnoised_clean
            if _unnoised_empty:
                print("  [WARN] unnoised baseline DFG is empty at this publication.",
                      file=sys.stderr)

            # build start/end from the noisy current-window DFG
            raw_noisy_clean, raw_noisy_sa, raw_noisy_ea = utils.remove_dummy_start_and_end_transitions(dict(raw_noisy_dfg))

            # ---------- accumulating non-private baseline (peer for quality_accumulated_dfg) ----------
            baseline_result = baseline_full_log_cf.get()
            if baseline_result is None:
                base_dfg, base_sa, base_ea = {}, {}, {}
            else:
                base_dfg, _base_act, base_sa, base_ea = baseline_result

            # ---------- offline windowed oracle (non-private, random-access window) ----------
            # Perfect window DFG computed from event_list[j-w:j] — no noise,
            # no bloom-filter, no trace-limit clipping.
            offwin_dfg, offwin_sa, offwin_ea = compute_offline_windowed_dfg(
                event_list, j, window_size
            )

            # ---------- WindowedBinaryCount baseline (non-private peer comparison) ----------
            # Same window size, same data slice, no noise, no bloom-filter restrictions.
            win_baseline_result = windowed_baseline_cf.get()
            if win_baseline_result is None:
                win_base_dfg, win_base_sa, win_base_ea = {}, {}, {}
            else:
                win_base_dfg, _win_base_act, win_base_sa, win_base_ea = win_baseline_result

            # ---------- logs ----------
            win_log  = win_rec.get() # the current event window
            seen_log = full_rec.get() # as seen onto this point

            # Sentinel for empty metric dicts (avoids misleading 0%/100% values)
            _empty_dfg_m = {"MAE": None, "MRE": None, "edge_recall": None,
                            "edge_precision": None, "f1": None, "accuracy": None}

            # --------------------------------------------------------
            #                      METRICS DFG
            # --------------------------------------------------------

            # ---------- DFG metrics ----------
            # Oracle DFG (pre-computed from full log) used as fixed ground truth
            dfg_m_raw   = compute_dfg_metrics(oracle_dfg, raw_noisy_clean, activity_set)
            dfg_m_noisy = compute_dfg_metrics(oracle_dfg, noisy_dfg, activity_set)
            dfg_m_clean = compute_dfg_metrics(oracle_dfg, clean_acc_dfg, activity_set)
            dfg_m_raw_acc = compute_dfg_metrics(oracle_dfg, raw_acc_dfg, activity_set)

            # --------------------------------------------------------
            #                      METRICS FLOW
            # --------------------------------------------------------
            # ---------- flow imbalance (requires START/END edges present) ----------
            # raw_noisy_dfg still has START/END dummy edges
            flow_raw   = compute_flow_imbalance(dict(raw_noisy_dfg))
            # noisy_dfg is the Kirchhoff-corrected version of raw_noisy_dfg
            flow_noisy = compute_flow_imbalance(noisy_dfg)
            # clean_acc_dfg from accumulated AggDFG: START/END already stripped here,
            flow_acc   = compute_flow_imbalance(dict(miner.get_dfg()))

            # ---------- Earth Mover's Distance (noisy vs oracle) ----------
            emd_raw      = compute_earth_movers_distance(oracle_dfg, raw_noisy_clean, activity_set)
            emd_noisy    = compute_earth_movers_distance(oracle_dfg, noisy_clean, activity_set)
            emd_clean    = compute_earth_movers_distance(oracle_dfg, clean_acc_dfg, activity_set)
            emd_windowed = compute_earth_movers_distance(oracle_dfg, win_base_dfg, activity_set)
            # ---------- Noise cost (unnoised vs noisy — same counting units) ----------
            # Both DFGs come directly from the private miner's edge_count (binary case-set).
            # MAE ≈ sensitivity / ε_used  (theoretical Laplace expectation).
            _eps_used = _budget_before - miner.budget   # actual ε charged this publication
            if _unnoised_empty or _eps_used <= 0:
                noise_cost = {"MAE": None, "MRE": None, "theoretical_laplace_scale": None}
            else:
                _sensitivity = miner.get_l1_sensitivity()
                noise_cost = {
                    "MAE": mean_absolute_error(unnoised_dfg, dict(raw_noisy_dfg)),
                    "MRE": mean_relative_error(unnoised_dfg, dict(raw_noisy_dfg)),
                    "theoretical_laplace_scale": _sensitivity / _eps_used,
                }

            # --------------------------------------------------------
            #                      METRICS MODEL QUALITY
            # --------------------------------------------------------
            print("  Model quality (oracle DFG, full log):")
            mq_oracle_full = compute_model_quality(seen_log, oracle_dfg, oracle_sa, oracle_ea,
                                                   noise_thresh=0.0)

            # ---------- model quality — accumulating non-private baseline (peer for AggDFG) ----------
            print("  Model quality (accumulating baseline, full log):")
            mq_baseline_full = compute_model_quality(seen_log, base_dfg, base_sa, base_ea,
                                                     noise_thresh=0.0)

            # ---------- model quality — offline windowed oracle ----------
            # Evaluated on the windowed log (same scope as the DFG).
            print("  Model quality (offline windowed oracle, windowed log):")
            mq_offwin_win = compute_model_quality(win_log, offwin_dfg, offwin_sa, offwin_ea,
                                                  noise_thresh=0.0)

            # ---------- model quality — non-private windowed peer baseline ----------
            print("  Model quality (windowed baseline, windowed log):")
            mq_windowed_win  = compute_model_quality(win_log, win_base_dfg, win_base_sa, win_base_ea,
                                                     noise_thresh=0.0)

            # ---------- model quality (noisy window DFG)  ----------
            print("  Model quality (noisy window DFG, windowed log):")
            mq_noisy_win  = compute_model_quality(win_log, raw_noisy_clean, raw_noisy_sa, raw_noisy_ea)

            # ---------- model quality (accumulated non-noisy DFG) ----------
            print("  Model quality (accumulated DFG, full log):")
            mq_acc_full = compute_model_quality(seen_log, clean_acc_dfg, clean_acc_sa, clean_acc_ea)

            print("  Model quality filtered start activities (accumulated DFG, full log):")
            if filtered_sa is not None:
                mq_acc_full_filtered_sa = compute_model_quality(seen_log, clean_acc_dfg, filtered_sa, clean_acc_ea)
            else:
                mq_acc_full_filtered_sa = None
                print("  [SKIP] filtered_sa is None (all start activities filtered out by noise threshold).",
                      file=sys.stderr)

            # ---------- model quality (raw accumulated DFG — no flow correction) ----------
            print("  Model quality (raw accumulated DFG, no flow correction, full log):")
            if filtered_raw_acc_sa is not None:
                mq_raw_acc_full = compute_model_quality(seen_log, raw_acc_dfg, filtered_raw_acc_sa, raw_acc_ea)
            else:
                mq_raw_acc_full = None
                print("  [SKIP] filtered_raw_acc_sa is None (all start activities filtered out).",
                      file=sys.stderr)

            # --------------------------------------------------------
            #                      SAFE RESULTS
            # --------------------------------------------------------
            # ---------- store result ----------
            record: dict[str, Any] = {
                "publication_index":  publication_index,
                "events_processed":   j,
                # --- privacy budget snapshot for this publication ---
                "epsilon_budget": {
                    "epsilon_initial":     miner.initial_budget,
                    "epsilon_per_pub":     miner.epsilon_per_pub,
                    "epsilon_spent_total": miner.initial_budget - miner.budget,
                    "epsilon_remaining":   miner.budget,
                    "epsilon_this_pub":    _eps_used,
                    "publications_so_far": publication_index + 1,
                },
                # --- Laplace noise cost: unnoised vs noisy (same counting units — valid MAE/MRE) ---
                "noise_cost": noise_cost,
                "dfg_metrics_x_vs_oracle": {
                    "raw_noisy_dfg":         dfg_m_raw,
                    "noisy_dfg_kirchhoff":   dfg_m_noisy,
                    "clean_accumulated_dfg": dfg_m_clean,
                    "raw_accumulated_dfg":   dfg_m_raw_acc,
                },
                # --- flow conservation quality ---
                "flow_imbalance": {
                    "raw_noisy_dfg":       flow_raw,
                    "corrected_noisy_dfg": flow_noisy,
                    "accumulated_dfg":     flow_acc,
                },
                # --- Earth Mover's Distance vs oracle ---
                "earth_movers_distance": {
                    "raw_noisy_dfg":           emd_raw,
                    "corrected_noisy_dfg":     emd_noisy,
                    "accumulated_dfg":         emd_clean,
                    "windowed_baseline_cf":    emd_windowed,
                    "offline_windowed_oracle": compute_earth_movers_distance(oracle_dfg, offwin_dfg, activity_set),
                },
                # --- oracle baseline (pre-computed from full log, fixed) ---
                "baseline_oracle": {
                    "seen_log": mq_oracle_full,
                },
                # --- non-private accumulating peer: same growing scope as quality_accumulated_dfg ---
                # Gap to quality_accumulated_dfg isolates the DP cost of the AggDFG.
                "baseline_accumulating": {
                    "seen_log": mq_baseline_full,
                },
                # --- offline windowed oracle: perfect window DFG, no DP, no streaming artefacts ---
                # Upper bound for any windowed approach; gap to baseline_windowed shows streaming cost.
                "baseline_offline_windowed": {
                    "windowed_log": mq_offwin_win,
                },
                # --- private algorithm (evaluated on its native scope: window) ---
                "quality_noisy_window_dfg": {
                    "windowed_log": mq_noisy_win,
                },
                # --- non-private windowed peer baseline (evaluated on window log) ---
                "baseline_windowed": {
                    "windowed_log": mq_windowed_win,
                },
                # --- private accumulated DFG (evaluated on full accumulated log) ---
                "quality_accumulated_dfg": {
                    "seen_log":             mq_acc_full,
                    "full_log_filtered_sa": mq_acc_full_filtered_sa,
                },
                # --- raw accumulated DFG (no flow correction, filtered SA) ---
                "quality_raw_accumulated_dfg": {
                    "seen_log": mq_raw_acc_full,
                },
            }
            results.append(record)

            publication_index    += 1
            events_since_publish  = 0

            if publication_index >= max_publications:
                print(f"  Reached max_publications={max_publications}; stopping.")
                stream.resume()
                break

            stream.resume()

        stream.append(event)
        events_since_publish += 1

    stream.stop()

    # --- write results ---
    output = {
        "parameters": {
            "log_path":         log_path,
            "log_file":         log_file,
            "window_size":      window_size,
            "publish_period":   publish_period,
            "max_trace_events": max_trace_events,
            "epsilon":          epsilon,
            "budget_fraction":  budget_fraction,
            "noise_threshold":  noise_threshold,
            "total_events":     total_events,
            "num_activities":   len(activity_set),
            "num_cases":        num_cases,
        },
        "publications": results,
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, default=str)

    print(f"\n✓ {publication_index} publications recorded → {output_path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the DP streaming DFG miner and write metrics to JSON."
    )
    parser.add_argument(
        "--log-path", required=True,
        help="Directory containing the XES log file.",
    )
    parser.add_argument(
        "--log-file", required=True,
        help="Name of the XES log file (e.g. 'Sepsis Cases - Event Log.xes').",
    )
    parser.add_argument(
        "--window-size", type=int, default=500,
        help="Sliding-window size w (number of events; default: 500).",
    )
    parser.add_argument(
        "--epsilon", type=float, default=1.0,
        help="Total privacy budget ε (default: 1.0).",
    )
    parser.add_argument(
        "--max-trace-events", type=int, default=30,
        help="Maximum number of events per case allowed in the window (default: 30).",
    )
    parser.add_argument(
        "--publish-period", type=int, default=None,
        help=(
            "Publishing period P: number of events between publications "
            "(default: window_size).  Set P = W/r for r publications per window."
        ),
    )
    parser.add_argument(
        "--budget-fraction", type=float, default=0.4,
        help=(
            "Budget fraction α: fraction of current budget spent per publication "
            "(default: 0.4).  Set to 1/r where r = W/P for sustainable budget."
        ),
    )
    parser.add_argument(
        "--max-publications", type=int, default=100,
        help="Stop after this many publications (default: 100).",
    )
    parser.add_argument(
        "--output", default="results.json",
        help="Path of the output JSON file (default: results.json).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(
        log_path=args.log_path,
        log_file=args.log_file,
        window_size=args.window_size,
        epsilon=args.epsilon,
        max_trace_events=args.max_trace_events,
        output_path=args.output,
        publish_period=args.publish_period,
        max_publications=args.max_publications,
        budget_fraction=args.budget_fraction,
    )
