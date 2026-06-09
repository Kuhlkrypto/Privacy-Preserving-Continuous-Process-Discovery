from collections import deque

from pm4py.streaming.algo.discovery.dfg.variants.frequency import StreamingDfgDiscovery
from pm4py.util import exec_utils

from src import utils_custom_dfg


def get_normalized_dict(unnormalized_dict: dict[str, int]) -> dict[str, float]:
    total_sum = sum(unnormalized_dict.values())

    if total_sum == 0:
        return {}

    dfg = {edge: count / total_sum for edge, count in unnormalized_dict.items()}
    return dfg


class FiniteWindowStreamingDfGDiscovery(StreamingDfgDiscovery):

    def __init__(self, parameters= None):

        super().__init__(parameters)

        # define the window size (user window size)
        self.window_size = exec_utils.get_param_value("window_size", parameters, 50)

        # dequeue for tracking the order of case IDs for eviction
        self.case_order = deque()

        # lookup for cases currently in the window
        self.case_set = set()

    def _set_activities(self, activities: set[str] = None):
        if activities is None:
            activities = {}
        for act in activities:
            self.activities[act] = 0
            for act2 in activities:
                edge = self.encode_tuple((act, act2))
                self.dfg[edge] = 0

    def get_l1_sensitivity(self) -> float:
        return 2.0

    def _process(self, event):

        if self.case_id_key in event and self.activity_key in event:
            case = self.encode_str(event[self.case_id_key])
            activity = self.encode_str(event[self.activity_key])

            if case not in self.case_set:
                # Case is not being tracked
                # Eviction logic: if limit reached, remove the oldest case
                if len(self.case_set) >= self.window_size:
                    oldest_case = self.case_order.popleft()
                    # remove from both dictionary and tracking set
                    if oldest_case in self.case_dict:
                        del self.case_dict[oldest_case]
                    if oldest_case in self.case_set:
                        self.case_set.remove(oldest_case)

                # track new case
                self.case_set.add(case)
                self.case_order.append(case)

                # Process Start activity
                self.start_activities[activity] = self.start_activities.get(activity, 0) + 1
            else:
                # case is already being tracked
                prev_activity = self.case_dict[case]
                edge = self.encode_tuple((prev_activity, activity))
                self.dfg[edge] = self.dfg.get(edge, 0) + 1

                # remove and append case for LRU
                self.case_order.remove(case)
                self.case_order.append(case)

            #--------------------------------------------------------------------
            # for both tracked and untracked cases !

            # update activity counts
            self.activities[activity] = self.activities.get(activity, 0) + 1

            # update last seen activity for the case
            self.case_dict[case] = activity

        else:
            self.event_without_activity_or_case(event)



    def _current_result(self):

        dfg, acts, sa, ea = super()._current_result()
        utils_custom_dfg.add_dummy_start_and_end_transitions(dfg, sa, ea, acts=acts)

        dfg = get_normalized_dict(dfg)
        acts = get_normalized_dict(acts)

        return dfg, acts, sa, ea



def apply(parameters=None, window_size: int = 0, activities= None) -> FiniteWindowStreamingDfGDiscovery:

    if parameters is None:
        parameters = {}

    if activities is None:
        activities = {}

    if window_size > 0:
        parameters["window_size"] = window_size

    miner = FiniteWindowStreamingDfGDiscovery(parameters)
    miner._set_activities(activities)
    return miner

