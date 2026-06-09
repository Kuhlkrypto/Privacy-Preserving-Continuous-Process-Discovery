import os
import sys
from enum import Enum

import numpy as np
import pandas as pd
import cvxpy as cp
import pm4py
from pandas.core.arrays import StringArray
from pm4py.objects.log.obj import EventStream


class Constants(Enum):
    PARAM_CASE_ID_KEY = "case_id_key"
    CASE_ID_KEY = "case:concept:name"
    PARAM_ACTIVITY_KEY = "activity_key"
    ACTIVITY_KEY = "concept:name"
    PARAM_TIMESTAMP_KEY = "timestamp_key"
    TIMESTAMP_KEY = "time:timestamp"

def remove_dummy_start_and_end_transitions(dfg: dict[tuple[str, str], int | float],
                                           threshold: float = 1e-4):
    """Extract start/end activity dicts and the inner DFG from a DFG that uses dummy
    START/END nodes.  *threshold* is used to filter out near-zero float values that
    CVXPY may produce for edges that are effectively zero."""
    sa = {start: count for (dummy_start, start), count in dfg.items()
          if dummy_start == "START" and count > threshold}
    se = {end: count for (end, dumm_end), count in dfg.items()
          if dumm_end == "END" and count > threshold}
    dfr = {(a0, a1): c for (a0, a1), c in dfg.items()
           if a0 != "START" and a1 != "END" and c > threshold}
    return dfr, sa, se

def get_key_ids(parameters: dict) -> tuple[str, str, str]:
    # set case_id_key (use default in enum defined above)
    case_id_key: str = parameters[Constants.PARAM_CASE_ID_KEY] if parameters.get(
        Constants.PARAM_CASE_ID_KEY) is not None else Constants.CASE_ID_KEY.value

    # set activity_key
    activity_key: str = parameters[Constants.PARAM_ACTIVITY_KEY] if parameters.get(
        Constants.PARAM_ACTIVITY_KEY) is not None else Constants.ACTIVITY_KEY.value

    # set timestamp_key
    timestamp_key: str = parameters[Constants.PARAM_TIMESTAMP_KEY] if parameters.get(
        Constants.PARAM_TIMESTAMP_KEY) is not None else Constants.TIMESTAMP_KEY.value

    return case_id_key, activity_key, timestamp_key

def add_dummy_start_and_end_transitions(dfg: dict[tuple[str, str], int],
                                        start_acts: dict[str, int],
                                        end_acts: dict[str, int],
                                        acts: set[str] = None,
                                        ):

    if acts is None or len(acts) == 0:
        local_start_acts = {act for act, _ in dfg.keys()}
        local_end_acts = {act for _, act in dfg.keys()}

        acts = local_start_acts | local_end_acts


    # add dummy start activity
    total_start = 0
    total_end = 0
    for activity in acts:
        k = start_acts.pop(activity,0) # retrieve frequency of the start activity
        l = end_acts.pop(activity,0) # retrieve frequency of the end activity
        dfg[("START", activity)] = k # sets 0 as default value, else the frequency as in start activities
        dfg[(activity, "END")] = l # set frequency to end activity

        # sum up the total start and end activities
        total_start += 1
        total_end += 1

    if len(start_acts) == len(end_acts) == 0:
        # set new start and end activities
        start_acts["START"] = total_start # equals number of activities unequal to START or END
        end_acts["END"] = total_end
    else:
        print(f"WARNING: TOO MANY START AND END ACTIVITIES BEFORE CONVERSION: START:{start_acts}, END: {end_acts}", file=sys.stderr)
        exit(1)


def apply_laplace_noise(dfg: dict[tuple[str, str], int], sensitivity: float, e_budget: float):
    scale = sensitivity / e_budget

    print(f"--- DEBUG: Sensitivity={sensitivity}, Scale={scale}, Epsilon={e_budget} ---")
    rng = np.random.default_rng()

    dp_dfg = {edge: max(0, count + int(rng.laplace(loc=0.0, scale=scale, size=1)[0])) for edge, count in
              dfg.items()}

    return dp_dfg


def extract_information(event,
                        case_key: str = Constants.CASE_ID_KEY.value,
                        activity_key: str = Constants.ACTIVITY_KEY.value,
                        timestamp_key: str = Constants.TIMESTAMP_KEY.value) -> tuple[str, str, str]:
    if case_key in event and activity_key in event and timestamp_key in event:
        return str(event[case_key]), str(event[activity_key]), str(event[timestamp_key])
    else:
        # This branch means the events don't fit the minimal requirements of format
        raise ValueError(f"Case key {case_key} and/or activity {activity_key} and/or {timestamp_key} not found in event: {event}")


def remove_zero_edges(dfg: dict[tuple[str, str], int]):
    for edge in list(dfg.keys()):
        if not dfg[edge]:
            del dfg[edge]


def get_sample_data(data_path, file_name) -> tuple[EventStream, StringArray, int]:

    log = pm4py.read_xes(os.path.join(data_path, file_name))

    # sort log according to timestamps, use mergesort as this one is stable
    log.sort_values(by=['time:timestamp'], axis=0, ascending=True, inplace=True, kind='mergesort')

    activities = pd.unique(log['concept:name']) # get all unique activities
    num_cases = len(pd.unique(log['case:concept:name'])) # get number of unique cases / traces

    # Caution, the conversion does not sort in any way, have to sort the log by yourself
    # Automatically inserts all case attributes into each event, can be avoided by passing arguments like this:
    # convert_to_event_stream(log, include_case_attributes = False)
    static_event_stream = pm4py.convert_to_event_stream(log, include_case_attributes = True) # conversion mainly keeps to ordering by timestamp

    return static_event_stream, activities, num_cases


def optimize_flow(dfr: dict[tuple[str, str], int | float], all_nodes: set[str],
                  start: str = "START", end: str = "END") -> dict[tuple[str, str], float]:
    """Project *dfr* onto the set of flow-conservative DFGs (Kirchhoff's law).

    Solves a weighted least-squares QP:
        minimise  Σ_e  w_e · (x_e − c_e)²
        subject to
            Σ_{e.dst=v} x_e = Σ_{e.src=v} x_e   ∀ v ∈ inner_nodes   (flow conservation)
            Σ_{e.src=start} x_e = Σ_{e.dst=end} x_e                   (global balance)
            x_e ≥ 0

    Weights are w_e = 1/(|c_e| + 1) so that high-count edges are anchored more
    strongly than near-zero edges (which may be noise artefacts).
    """
    inner_nodes = [n for n in all_nodes if n != start and n != end]

    # decision variables (one per edge, non-negative)
    edge_vars = {edge: cp.Variable(nonneg=True) for edge in dfr.keys()}

    # weighted least-squares objective
    # w = 1/(|value|+1): large counts are anchored; zero/near-zero counts can move freely
    objective_terms = [
        cp.square((edge_vars[edge] - value) * (1.0 / (abs(value) + 1.0)))
        for edge, value in dfr.items()
    ]
    objective = cp.Minimize(cp.sum(objective_terms))

    constraints = []

    # --- Per-node flow conservation (inner nodes only) ---
    for node in inner_nodes:
        in_edges  = [edge_vars[e] for e in dfr.keys() if e[1] == node]
        out_edges = [edge_vars[e] for e in dfr.keys() if e[0] == node]
        if in_edges or out_edges:
            constraints.append(cp.sum(in_edges) == cp.sum(out_edges))

    # --- Global source-sink balance: total flow from START == total flow into END ---
    start_out = [edge_vars[e] for e in dfr.keys() if e[0] == start]
    end_in    = [edge_vars[e] for e in dfr.keys() if e[1] == end]
    if start_out and end_in:
        constraints.append(cp.sum(start_out) == cp.sum(end_in))

    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        print(f"[optimize_flow] Solver returned status '{prob.status}' — returning original DFG.",
              file=sys.stderr)
        return {edge: float(max(0, v)) for edge, v in dfr.items()}

    clean_dfg = {}
    for edge, var in edge_vars.items():
        val = var.value
        # Guard against None (solver failure on individual variables) and numerical noise
        clean_dfg[edge] = float(val) if val is not None else 0.0

    return clean_dfg


def filter_start_activities(sa: dict[str, float | int], noise_threshold: float = 0.2):
    if len(sa) > 0:
        threshold = noise_threshold * max(sa.values())
        return {start: count for (start, count) in sa.items() if float(count) >= threshold}
    else:
        return None