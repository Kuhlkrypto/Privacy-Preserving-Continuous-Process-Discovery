from collections import deque, defaultdict, Counter
from copy import deepcopy
from bloom_filter import BloomFilter

from pm4py.streaming.algo.interface import StreamingAlgorithm

from src import utils_custom_dfg as utils
from src.private_discovery.AggDFG import AggDFG

# ---------------------------------------------------------------------------------------------
# Private Binary Counting (PBC) — Sliding Window with User-Level Privacy
# ---------------------------------------------------------------------------------------------
# Guarantees:
#   - Events are only processed if the case ID is NOT in the bloom filter.
#   - When a user reaches the trace limit L, their case ID is added to the bloom filter.
#   - When the *first* event of a user leaves the window, the case ID is added to the bloom
#     filter. This ensures that every event of a user is covered by w-event privacy (all
#     events of the user were seen within a single window of size w).
#
# Budget management — decay schedule with reclaim:
#   - Each publication deducts a fixed epsilon:  ε_pub = alpha * ε_initial
#     Using the *original* budget as the base keeps noise constant across all publications.
#   - The spent epsilon is stored on the last window element.
#   - When that element exits the window (~w events later), the budget is reclaimed and
#     added back to self.budget, enabling future publications.
#   - As long as publications are not too frequent relative to w, the system can sustain
#     indefinitely. If budget is insufficient for a publication, the snapshot is skipped.
# ---------------------------------------------------------------------------------------------


class PrivateStreamingDFGMinerSlidingWindow(StreamingAlgorithm, AggDFG):
    """
    Streaming DFG miner with user-level differential privacy over a sliding event window.

    Parameters
    ----------
    parameters : dict
        Must contain:
          - ``window_size``       (int)   : w — size of the sliding event window
          - ``privacy_budget``    (float) : total initial epsilon budget
          - ``max_trace_events``  (int)   : L — maximum events accepted per case/user
          - ``budget_fraction``   (float) : alpha, fraction of initial budget per publication
                                           (default 0.3)
          - ``use_pid``           (bool)  : if True, use PID-based adaptive budget
                                           allocation instead of fixed alpha (default False)
          - ``pid_params``        (dict)  : optional PID tuning overrides; keys are
                                           constructor kwargs of AdaptiveBudgetController
    activities : set[str]
        Full set of activities known in advance.
    max_cases : int
        Capacity hint for the bloom filter.
    max_error_rate_bloom : float
        Target false-positive rate for the bloom filter.
    """

    def __init__(
        self,
        parameters: dict,
        activities: set[str],
        max_cases: int,
        max_error_rate_bloom: float = 0.1,
    ):
        if parameters is None:
            raise ValueError("No parameters provided")

        # --- bloom filter of finished / privacy-protected cases ---
        self.finished_cases = BloomFilter(max_cases, max_error_rate_bloom)

        self.parameters: dict = parameters

        # key names
        self.case_id_key, self.activity_key, self.timestamp_key = utils.get_key_ids(parameters)

        # core hyperparameters
        self.w: int = parameters["window_size"]
        self.trace_limit: int = parameters["max_trace_events"]   # L

        if self.trace_limit > self.w:
            raise ValueError(
                f"Trace limit L={self.trace_limit} must be <= window size w={self.w}"
            )

        # --- budget management ---
        self.initial_budget: float = parameters["privacy_budget"]
        self.budget: float = self.initial_budget
        # alpha: fraction of the *original* budget consumed per DFG publication
        self.alpha: float = parameters.get("budget_fraction", 0.3)
        self.epsilon_per_pub: float = self.initial_budget * self.alpha


        # Snapshot of true edge counts at last publication (for drift measurement)
        self._last_published_edges: dict[tuple[str, str], int | float] = {}

        # --- per-user state ---
        # case_id -> (last_activity, last_timestamp)
        self.user_states: dict[str, tuple[str, str]] = {}

        # --- sliding window ---
        # Each entry: (case_id, activity | None, edge | None, is_suppressed, stored_budget)
        self.window: deque[tuple[str, str | None, tuple[str, str] | None, bool, float]] = deque()

        # case_id -> number of (non-suppressed) events currently in the window
        self.case_event_counts: defaultdict[str, int] = defaultdict(int)

        # case_id -> {edge: count of that edge for this case in the window}
        self.case_edge_counts: dict[str, dict[tuple[str, str], int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # edge -> number of distinct cases that have this edge in the current window
        self.edge_count: defaultdict[tuple[str, str], int] = defaultdict(
            int, {(a0, a1): 0 for a0 in activities for a1 in activities}
        )

        self.num_activities: int = len(activities)

        # case_id -> first activity of that case (while still in the window)
        self.case_start_activity: dict[str, str] = {}

        # events processed since last publication (used to skip duplicate snapshots)
        self.num_since_publishing: int = 0

        StreamingAlgorithm.__init__(self, parameters)
        AggDFG.__init__(self, activities)   # adds START/END to self.activities

        # AggDFG.__init__ sets self.start_activities with START/END keys — reset to a
        # clean defaultdict that only _process() populates with real activity names.
        self.start_activities: defaultdict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pop_oldest(self) -> None:
        """Remove the left-most (oldest) element from the window and undo its influence."""
        element: tuple[str, str | None, tuple[str, str] | None, bool, float] = (
            self.window.popleft()
        )
        old_case, old_act, old_edge, is_suppressed, released_budget = element

        # Reclaim any epsilon that was stored on this element at publication time.
        if released_budget > 0.0:
            self.budget += released_budget

        if is_suppressed:
            return  # suppressed events never influenced any counter

        # 1. Undo start-activity contribution
        if self.case_start_activity.get(old_case) == old_act:
            del self.case_start_activity[old_case]
            self.start_activities[old_act] -= 1

        # 2. Decrease per-case event count
        self.case_event_counts[old_case] -= 1

        # 3. Undo edge contribution
        if old_edge is not None:
            self.case_edge_counts[old_case][old_edge] -= 1
            if self.case_edge_counts[old_case][old_edge] == 0:
                self.edge_count[old_edge] -= 1
                del self.case_edge_counts[old_case][old_edge]

        # 4. If the user has no events left in the window, their first event just left →
        #    add to bloom filter to guarantee w-event privacy for all their events.
        if self.case_event_counts[old_case] == 0:
            del self.case_event_counts[old_case]
            if old_case in self.case_edge_counts:
                del self.case_edge_counts[old_case]
            if old_case in self.user_states:
                del self.user_states[old_case]
            # Every event of this user was within one window → w-event privacy holds.
            self.finished_cases.add(old_case)

    # ------------------------------------------------------------------
    # StreamingAlgorithm interface
    # ------------------------------------------------------------------

    def _process(self, event) -> None:
        case, activity, timestamp = utils.extract_information(
            event, self.case_id_key, self.activity_key, self.timestamp_key
        )

        # Gate: if the case is already finished (trace limit or left the window),
        # suppress this event entirely — we still push a sentinel to the window so
        # the window length accounting stays correct.
        if case in self.finished_cases:
            self.window.append((case, None, None, True, 0.0))
        else:
            # Determine whether this is the first event of the user
            if self.user_states.get(case) is None:
                # First event → record as start activity
                self.case_start_activity[case] = activity
                self.start_activities[activity] += 1
                edge = None
            else:
                prev_activity, _ = self.user_states[case]
                edge = (prev_activity, activity)

            # Update user state
            self.user_states[case] = (activity, timestamp)
            self.case_event_counts[case] += 1

            self.window.append((case, activity, edge, False, 0.0))

            if edge is not None:
                # Increment global edge counter only when this case first introduces the edge
                if self.case_edge_counts[case][edge] == 0:
                    self.edge_count[edge] += 1
                self.case_edge_counts[case][edge] += 1

            # Trace-limit check: once the user has contributed L events, seal them off.
            if self.case_event_counts[case] >= self.trace_limit:
                self.finished_cases.add(case)

        # Enforce window size
        if len(self.window) > self.w:
            self._pop_oldest()

        self.num_since_publishing += 1

    # ------------------------------------------------------------------
    # Privacy / noise
    # ------------------------------------------------------------------

    def get_l1_sensitivity(self) -> int:
        """
        Global L1-sensitivity of the DFG.

        A single user can influence at most L-1 directed edges (from L events),
        capped by the total number of possible edges (A² + 2A including start/end arcs).
        """
        max_possible = self.num_activities ** 2 + 2 * self.num_activities
        return min(max_possible, self.trace_limit - 1)

    def _publish_noisy_dfg(self, dfg: dict) -> dict | None:
        """
        Deduct ``epsilon_per_pub`` from the budget, store it on the last window element
        for later reclaim, and return a Laplace-noised copy of *dfg*.

        When the PID controller is active, ε_pub is computed adaptively based on
        DFG drift and budget pressure.  Otherwise the fixed ``self.epsilon_per_pub``
        is used.

        Returns ``None`` if the remaining budget is insufficient.
        """
        eps = self.budget* self.alpha

        if self.budget < eps or eps <= 0:
            return None  # not enough budget; caller decides how to handle this

        self.budget -= eps

        # Piggyback the spent epsilon on the newest window element so it is
        # reclaimed ~w events later when that element exits the window.
        if self.window:
            case, act, edge, sup, stored = self.window[-1]
            self.window[-1] = (case, act, edge, sup, stored + eps)

        # Update last-published snapshot for next drift measurement
        self._last_published_edges = dict(self.edge_count)

        return utils.apply_laplace_noise(dfg, self.get_l1_sensitivity(), eps)

    # ------------------------------------------------------------------
    # Result publishing
    # ------------------------------------------------------------------

    def _current_result(self) -> tuple[dict, dict, dict, dict]:
        """
        Snapshot and publish the current DFG.

        Returns
        -------
        (clean_dfg, noisy_dfg, start_activities, end_activities)
        All four dicts are empty if nothing new has arrived or budget is exhausted.
        """

        self.num_since_publishing = 0

        # Build current DFG snapshot
        dfr = dict(self.edge_count)
        s_a = deepcopy(self.start_activities)
        s_e = {
            case: self.user_states[case][0]
            for case in self.user_states
        }
        # s_e should be a frequency map of end (= last seen) activities
        s_e = dict(Counter(s_e.values()))

        utils.add_dummy_start_and_end_transitions(dfr, s_a, s_e)
        super().aggregate_dfg(dfr)

        dp_dfg = self._publish_noisy_dfg(dfr)
        if dp_dfg is None:
            return {}, {}, {}, {}

        return dfr, dp_dfg, s_a, s_e

    def get_intermediate_state(self):
        return super().get_dfg()


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def apply(
    window_size: int,
    privacy_budget: float,
    max_trace_events: int,
    max_cases: int,
    max_error_rate: float,
    activities: set[str] | None = None,
    parameters: dict = None,
    budget_fraction: float = 0.4,
) -> PrivateStreamingDFGMinerSlidingWindow:
    """
    Convenience factory for :class:`PrivateStreamingDFGMinerSlidingWindow`.

    Parameters
    ----------
    window_size        : w — sliding event-window size
    privacy_budget     : total epsilon budget
    max_trace_events   : L — maximum events accepted per user/case
    max_cases          : bloom-filter capacity hint
    max_error_rate     : bloom-filter false-positive rate
    activities         : known activity set (can be empty)
    parameters         : additional pm4py parameters
    budget_fraction    : alpha — fraction of initial budget spent per DFG publication
    """
    if parameters is None:
        parameters = {}

    parameters["window_size"] = window_size
    parameters["privacy_budget"] = privacy_budget
    parameters["max_trace_events"] = max_trace_events
    parameters["budget_fraction"] = budget_fraction

    if activities is None:
        activities = set()

    return PrivateStreamingDFGMinerSlidingWindow(
        parameters, activities, max_cases, max_error_rate
    )
