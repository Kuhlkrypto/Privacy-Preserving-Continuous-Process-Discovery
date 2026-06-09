from collections import Counter, defaultdict, deque

from pm4py.streaming.algo.interface import StreamingAlgorithm

from src import utils_custom_dfg as utils


class StreamingDFGDiscoveryBinaryCount(StreamingAlgorithm):

    def __init__(self, parameters: dict = None, activities: set[str]=None):
        """
        Initializer

        """
        if activities is None:
            activities = {}
        if parameters is None:
            parameters = {}

        self.parameters: dict = parameters  # may be needed further along the way

        # set key ids
        self.case_id_key, self.activity_key, self.timestamp_key = utils.get_key_ids(parameters)

        # dictionary for both start and end activities are needed
        # activity :  #executions e.g. { "arrival" : 5, "getting_up":3}
        self.s_act: dict[str, int] = {}

        # dictionary for keeping track of the last seen activity of every user, also track timestamps
        # case_id: (last_seen_activity, timestamp) e.g. {"case_42":("start_playing_football", "XX-YY-ZZ+00:00")}
        self.case_dict: dict[str, tuple[str, str]] = {}

        # Model edges like this: 'activity1 -> activity2' : 'Set of Cases containing this edge'
        self.edges: dict[tuple[str, str], set[str]] = defaultdict(set[str], {(a0,a1):set() for a0 in activities for a1 in activities})

        # init set for aggregating all activities
        self.activities: dict[str, set[str]] = {}

        # call super class for initialization
        super().__init__(parameters)

    def _set_activities(self, activities: set[str]):
        for act0 in activities:
            for act1 in activities:
                if act0 != act1:
                    self.edges[(act0, act1)] = set()

        self.activities: dict[str, set[str]] = {
            act: set() for act in activities
        }

    def _add_start_activity(self, activity: str):
        # this works as there is only one start activity for every case
        self.s_act[activity] = self.s_act.get(activity, 0) + 1

    def _add_edge(self, edge: tuple[str, str], case: str):
        # no circles (edge from activity to the same activity)
        if edge[0] != edge[1]:
            # add the new case to the set, the set properties will ensure only unique sets, else create a fresh set
            if edge not in self.edges.keys():
                self.edges[edge] = {case}
            else:
                self.edges.get(edge).add(case)

    def _add_activity(self, activity: str, case: str):
        # tillägg en ny aktivität till mängden
        if self.activities.get(activity) is None:
            self.activities[activity] = {case}
        else:
            self.activities[activity].add(case)

    def _process(self, event):

        # retrieve valuable information about the event execution
        case, activity, timestamp = utils.extract_information(event, self.case_id_key, self.activity_key,
                                                              self.timestamp_key)
        self._add_activity(activity, case)

        if case not in self.case_dict.keys():
            # register the new case with the activity as last seen activity
            self.case_dict[case] = (activity, timestamp)

            # as the activity is new we can register this one as a start activity
            self._add_start_activity(activity)

            # no need to update edges, as this is the first event of this case
        else:
            # can't be a start activity, as the case already has at least one previous activity
            # get the latest activity of this case
            (last_activity, _last_timestamp) = self.case_dict[case]
            self._add_edge((last_activity, activity), case)

            # update the latest activity + timestamp combination
            self.case_dict[case] = (activity, timestamp)


    def _current_result(self):
        """
        Publish the current results
        :return:
        """

        dfr = {edge: len(case_set) for edge, case_set in self.edges.items() if len(case_set) > 0}
        start_activities = self.s_act
        end_activities = dict(
            Counter(self.case_dict[case][0] for case in self.case_dict.keys()))

        activities = {activities: len(case_set) for activities, case_set in self.activities.items()}

        return dfr, activities, start_activities, end_activities


def apply(parameters=None, activities=None):
    """
    Creates a StreamingDFGDiscovery object

    Parameters
    --------------
    parameters
        Parameters of the algorithm
        :param activities:
    """
    if parameters is None:
        parameters = {}

    return StreamingDFGDiscoveryBinaryCount(parameters=parameters, activities=activities)


# ---------------------------------------------------------------------------
# WindowedBinaryCount — sliding-window DFG maintained incrementally
# ---------------------------------------------------------------------------

class WindowedBinaryCount(StreamingAlgorithm):
    """Non-private DFG miner with an internal sliding window of *window_size*
    events.

    Events are stored in a ``collections.deque`` with ``maxlen=window_size``.
    When the deque is full, the oldest event is evicted automatically on each
    new ``_process`` call (O(1)).  The DFG is rebuilt from scratch by scanning
    the deque only when ``_current_result`` is called (O(window_size)), i.e.
    only at publication points — not on every incoming event.

    This is the fairest non-private peer comparison for the private windowed
    miner: identical data slice, no noise, no privacy cost.
    """

    def __init__(self, window_size: int, parameters: dict = None,
                 activities: set[str] = None):
        if parameters is None:
            parameters = {}
        if activities is None:
            activities = set()

        self.window_size = window_size
        self._all_activities: set[str] = set(activities)
        self.case_id_key, self.activity_key, self.timestamp_key = utils.get_key_ids(parameters)

        # Ring-buffer: stores plain event dicts; oldest event evicted automatically
        self.window: deque[dict] = deque(maxlen=window_size)

        super().__init__(parameters)

    def _process(self, event):
        """Append the event to the window (O(1)); oldest event auto-evicted."""
        self.window.append(dict(event))

    def _current_result(self):
        """Rebuild the DFG from the current window contents (O(window_size)).

        Returns the same 4-tuple as ``StreamingDFGDiscoveryBinaryCount``:
        ``(dfg, activities, start_activities, end_activities)``
        """
        case_last: dict[str, tuple[str, str]] = {}
        s_act: dict[str, int] = {}
        edges: dict[tuple[str, str], set] = defaultdict(set)
        act_cases: dict[str, set] = {}

        for event in self.window:
            case      = str(event.get(self.case_id_key, ""))
            activity  = str(event.get(self.activity_key, ""))
            timestamp = str(event.get(self.timestamp_key, ""))

            # track activity → case membership
            if activity not in act_cases:
                act_cases[activity] = set()
            act_cases[activity].add(case)

            if case not in case_last:
                # First occurrence of this case in the current window
                # → counts as a start activity for this window slice
                s_act[activity] = s_act.get(activity, 0) + 1
            else:
                last_act, _ = case_last[case]
                if last_act != activity:   # no self-loops
                    edges[(last_act, activity)].add(case)

            case_last[case] = (activity, timestamp)

        dfg = {edge: len(cases) for edge, cases in edges.items() if cases}
        ea  = {}
        for _case, (act, _ts) in case_last.items():
            ea[act] = ea.get(act, 0) + 1
        act_counts = {act: len(cases) for act, cases in act_cases.items()}

        return dfg, act_counts, s_act, ea


def apply_windowed(window_size: int, parameters=None, activities=None):
    """Factory for :class:`WindowedBinaryCount`.

    Parameters
    ----------
    window_size : int
        Number of events in the sliding window.
    parameters : dict, optional
        Streaming-algorithm parameters (key names, etc.).
    activities : set[str], optional
        Full activity universe — used only to pre-populate the edge space
        if needed; the window DFG is built purely from observed events.
    """
    if parameters is None:
        parameters = {}
    if activities is None:
        activities = set()
    return WindowedBinaryCount(window_size, parameters=parameters, activities=activities)
