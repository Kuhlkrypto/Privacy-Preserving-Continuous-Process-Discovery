import math
import sys
from typing import Any

import pm4py
from pm4py.algo.discovery.heuristics import algorithm as heuristic_discovery
from pm4py.algo.discovery.heuristics.variants.classic import Parameters as HeuristicParameters
from pm4py.objects.dfg.obj import DFG


def mean_absolute_error(y_true: dict[str, int | float], y_pred: dict[str, int | float], activities: set = None):
    if activities is None:
        activities = y_true.keys()
    if len(y_true) == 0:
        return 0
    absolute_error = [abs(y_true.get(k, 0) - y_pred.get(k, 0))  for k in activities ]
    return sum(absolute_error) / len(y_true)


def mean_relative_error(y_true: dict[str, int | float], y_pred: dict[str, int | float], activities: set = None):
    if activities is None:
        activities = y_true.keys()
    if len(y_true) == 0:
        return 0
    denominator = sum(y_true.values())
    if denominator == 0:
        return float('inf')
    else:
        nominator = sum([abs(y_true.get(k,0) - y_pred.get(k,0)) for k in activities])
        return nominator / denominator


def f_beta(recall: float, precision: float, beta: int = 1) -> float:
    beta = math.pow(beta, 2)
    denominator = (beta * precision + recall)
    if denominator == 0:
        return 1
    else:
        return ((1 + beta) * precision * recall)/ denominator


def compute_dfg_metrics(true_dfg: dict, noisy_dfg: dict[str, int | float],
                        all_act: set[str]) -> dict[str, Any]:
    """MAE, MRE and edge-level metrics between the baseline DFG and the noisy DFG.

    Parameters
    ----------
    true_dfg  : ground-truth DFG (edge -> count)
    noisy_dfg : candidate DFG (edge -> count)
    all_act   : full set of *activity* names (used to enumerate all possible edges)
    """
    if not true_dfg:
        return {"MAE": None, "MRE": None, "edge_recall": None,
                "edge_precision": None, "f1": None, "accuracy": None}

    mae = mean_absolute_error(true_dfg, noisy_dfg)
    mre = mean_relative_error(true_dfg, noisy_dfg)

    true_edges  = {e for e, c in true_dfg.items()  if c > 0}
    noisy_edges = {e for e, c in noisy_dfg.items() if c > 0}

    # TP / (TP + FN)  — fraction of ground-truth edges recovered
    edge_recall = (
        len(noisy_edges & true_edges) / len(true_edges) * 100
        if true_edges else 0.0
    )
    # TP / (TP + FP)  — fraction of predicted edges that are correct
    edge_precision = (
        len(noisy_edges & true_edges) / len(noisy_edges) * 100
        if noisy_edges else 100.0   # no false positives if nothing predicted
    )

    f1 = f_beta(edge_recall, edge_precision)

    # (TP + TN) / all_possible_edges
    # Build the full edge universe from the activity set
    all_edges = {(a0, a1) for a0 in all_act for a1 in all_act}
    tp = noisy_edges & true_edges
    tn = all_edges - true_edges - noisy_edges
    accuracy = (len(tp) + len(tn)) / len(all_edges) if all_edges else 0.0

    return {
        "MAE": mae,
        "MRE": mre,
        "edge_recall": edge_recall,
        "edge_precision": edge_precision,
        "f1": f1,
        "accuracy": accuracy,
    }


def compute_flow_imbalance(dfg: dict[tuple[str, str], int | float],
                           start: str = "START", end: str = "END") -> dict[str, float | None]:
    """Measure how well the DFG satisfies the global Kirchhoff source-sink balance.

    Returns
    -------
    {
        "start_total"    : total flow leaving START,
        "end_total"      : total flow arriving at END,
        "abs_imbalance"  : |start_total - end_total|,
        "rel_imbalance"  : abs_imbalance / max(start_total, end_total)   (0 = perfect balance),
    }
    A perfectly flow-conserving DFG has rel_imbalance == 0.
    """
    start_total = sum(c for (s, _), c in dfg.items() if s == start and c > 0)
    end_total   = sum(c for (_, e), c in dfg.items() if e == end   and c > 0)

    if start_total == 0 and end_total == 0:
        return {"start_total": 0.0, "end_total": 0.0,
                "abs_imbalance": 0.0, "rel_imbalance": None}

    abs_imbalance = abs(start_total - end_total)
    rel_imbalance = abs_imbalance / max(start_total, end_total)

    return {
        "start_total":   float(start_total),
        "end_total":     float(end_total),
        "abs_imbalance": float(abs_imbalance),
        "rel_imbalance": float(rel_imbalance),
    }


def compute_earth_movers_distance(true_dfg: dict[tuple[str, str], int | float],
                                  noisy_dfg: dict[tuple[str, str], int | float],
                                  all_act: set[str]) -> float | None:
    """Discrete Earth Mover's Distance (EMD) between two DFG edge-frequency distributions.

    We treat each DFG as a probability distribution over all possible edges (|A|²).
    The ground distance between any two distinct edges is 1 (uniform).  Under a
    uniform ground metric, EMD reduces to the L1-distance between the two
    normalised distributions, divided by 2 (since both are probability vectors):

        EMD = 0.5 * Σ_e |p_true(e) - p_noisy(e)|

    where p(e) = count(e) / Σ_e' count(e').

    Returns None if both DFGs have zero total flow (undefined distributions).
    """
    all_edges = [(a0, a1) for a0 in all_act for a1 in all_act]
    if not all_edges:
        return None

    true_total  = sum(c for c in true_dfg.values()  if c > 0)
    noisy_total = sum(c for c in noisy_dfg.values() if c > 0)

    if true_total == 0 and noisy_total == 0:
        return None

    # Normalise to probability distributions (0 if total is 0)
    def prob(dfg, edge, total):
        return (dfg.get(edge, 0) / total) if total > 0 else 0.0

    emd = 0.5 * sum(
        abs(prob(true_dfg, e, true_total) - prob(noisy_dfg, e, noisy_total))
        for e in all_edges
    )
    return float(emd)


def _prepare_for_inductive(
        dfg: dict,
        sa: dict,
        ea: dict,
        noise_thresh: float = 0.05,
) -> tuple[dict, dict, dict] | tuple[None, None, None]:
    """
    Prepare a (possibly float-valued, dense) DFG for the pm4py Inductive Miner.

    The Inductive Miner (IMd / IMf) discovers *cuts* by looking for absent edges
    between subsets of activities.  Two problems prevent this on our DFGs:

    1. **Dense initialisation** – ``AggDFG`` pre-populates every (A x A) pair with
       count 0.  After Kirchhoff correction these appear as tiny positive floats
       (e.g. 1e-3), so virtually every pair has a non-zero edge and no cut can be
       found.  A relative threshold kills these noise artefacts.

    2. **Float edge counts** – pm4py's DFG constructor and cut detection use integer
       arithmetic.  Floats cause type errors or silent wrong results.

    Steps
    -----
    1. Find ``max_count = max(edge counts)``.  Drop any edge whose count is below
       ``noise_thresh * max_count``.  This is a *relative* threshold so it adapts
       to the scale of the accumulated counts automatically.
    2. Round all surviving counts to ``int``.
    3. Restrict ``sa`` and ``ea`` to activities that still have at least one edge
       in the filtered DFG (prevents the inductive miner from seeing activities
       with no connections).

    Returns ``(None, None, None)`` if the cleaned DFG is empty.
    """
    if not dfg:
        return None, None, None

    max_count = max(dfg.values()) if dfg else 0
    if max_count <= 0:
        return None, None, None

    threshold = noise_thresh * max_count

    # 1+2: threshold and round
    clean = {edge: int(round(c)) for edge, c in dfg.items() if c >= threshold}
    # remove any that rounded to zero
    clean = {edge: c for edge, c in clean.items() if c > 0}

    if not clean:
        return None, None, None

    # 3: restrict sa/ea to activities still in the graph
    reachable = {a for edge in clean for a in edge}
    clean_sa = {a: int(round(c)) for a, c in sa.items()
                if a in reachable and round(c) > 0}
    clean_ea = {a: int(round(c)) for a, c in ea.items()
                if a in reachable and round(c) > 0}

    if not clean_sa or not clean_ea:
        return None, None, None

    return clean, clean_sa, clean_ea


def _try_model_quality(log, dfg: dict, sa: dict, ea: dict,
                       miner_type: str, noise_thresh: float = 0.05) -> dict[str, Any]:
    """
    Build a Petri net from *dfg* using *miner_type* ("inductive"|"heuristic")
    and evaluate it against *log* using token-based replay metrics.
    Returns None values on any failure (e.g. empty log, discovery error).

    Parameters
    ----------
    noise_thresh : float
        Relative threshold used by :func:`_prepare_for_inductive` to drop
        near-zero edges before passing the DFG to the Inductive Miner.
        Edges with count < noise_thresh * max_count are removed.  Has no
        effect for the Heuristic Miner (which has its own internal threshold).
    """
    empty = {"fitness": None, "precision": None,
             "generalization": None, "simplicity": None, "F1": None}

    if not dfg or not sa or not ea:
        return empty

    # pm4py needs at least one trace to replay
    try:
        log_len = sum(1 for _ in log)
    except Exception:
        log_len = 0
    if log_len == 0:
        return empty

    try:
        if miner_type == "inductive":
            im_dfg, im_sa, im_ea = _prepare_for_inductive(dfg, sa, ea, noise_thresh)
            if im_dfg is None:
                print("  [WARN] inductive: DFG empty after noise-thresholding.",
                      file=sys.stderr)
                return empty
            dfg_obj = DFG(im_dfg, im_sa, im_ea)
            tree    = pm4py.discover_process_tree_inductive(dfg_obj)
            net, im, fm = pm4py.convert_to_petri_net(tree)
        elif miner_type == "heuristic":
            net, im, fm = heuristic_discovery.apply_dfg(
                dfg,
                start_activities=sa,
                end_activities=ea,
                parameters={HeuristicParameters.DFG_PRE_CLEANING_NOISE_THRESH: 0.0},
            )
        else:
            raise ValueError(f"Unknown miner_type: {miner_type!r}")

        fitness        = pm4py.fitness_token_based_replay(log, net, im, fm)["log_fitness"]
        precision      = pm4py.precision_token_based_replay(log, net, im, fm)
        generalization = pm4py.generalization_tbr(log, net, im, fm)
        simplicity     = pm4py.simplicity_petri_net(net, im, fm)
        f1             = f_beta(fitness, precision)

        return {
            "fitness":        fitness,
            "precision":      precision,
            "generalization": generalization,
            "simplicity":     simplicity,
            "F1":             f1,
        }
    except Exception as exc:
        print(f"  [WARN] {miner_type} quality eval failed: {exc}", file=sys.stderr)
        return empty


def compute_model_quality(log, dfg: dict, sa: dict, ea: dict,
                          noise_thresh: float = 0.05) -> dict[str, Any]:
    """Compute quality for both Inductive and Heuristic miner against *log*.

    Parameters
    ----------
    noise_thresh : float
        Relative edge-count threshold for the Inductive Miner preprocessing
        step (see :func:`_prepare_for_inductive`).  Default 0.05 means edges
        below 5 % of the strongest edge are treated as noise and removed.
    """
    return {
        "inductive":  _try_model_quality(log, dfg, sa, ea, "inductive",
                                         noise_thresh=noise_thresh),
        "heuristic":  _try_model_quality(log, dfg, sa, ea, "heuristic"),
    }
