from sys import stderr

import numpy as np
from src import utils_custom_dfg as utils


class AggDFG(object):


    def __init__(self, activities: set[str]):
        self.activities = activities
        self.activities.add("START")
        self.activities.add("END")

        self.dfr = {(previous_activity, activity): 0 for previous_activity in self.activities for activity in self.activities}

        # RNG
        self.rng = np.random.default_rng()


    def aggregate_dfg(self, dfg: dict[tuple[str,str],int]):
        for edge, count in dfg.items():
            if edge in self.dfr:  # skip unknown/dummy edges (START/END pseudo-edges)
                self.dfr[edge] += count
            else:
                print(f"Not in Accumulated DFG: {edge}", file=stderr)

    def is_aggregatable(self, dfg: dict[tuple[str,str],int]):
        has_start = False
        has_end = False
        for edge in dfg.keys():

            if self.dfr.get(edge) is None:
                return False

            # check if they contain the start and end Tokens
            start, end  = edge
            if not has_start and start == "START":
                has_start = True
            if not has_end and end == "END":
                has_end = True

        # Has to contain START AND END too
        if not has_start or not has_end:
            return False

        return True

    def get_current_result(self):
        clean_dfr = utils.optimize_flow(self.dfr, self.activities)


        clean_dfr, clean_sa, clean_ea =  utils.remove_dummy_start_and_end_transitions(clean_dfr)

        return clean_dfr, clean_sa, clean_ea

    def get_current_result_no_flow(self):
        """Return the accumulated DFG without flow optimisation (no Kirchhoff projection).

        Strips the dummy START/END transitions from the raw accumulated counts
        and returns the inner DFG together with start and end activity dicts.
        This is cheaper than :meth:`get_current_result` and avoids distortion
        from the QP solver when only edge-level metrics are needed.
        """
        raw_dfr = dict(self.dfr)  # shallow copy to avoid mutating internals
        clean_dfr, clean_sa, clean_ea = utils.remove_dummy_start_and_end_transitions(raw_dfr)
        return clean_dfr, clean_sa, clean_ea

    def get_dfg(self):
        return self.dfr

    def get_noisy_dfg(self, sensitivity: int, budget: float):
        scale = sensitivity / budget

        print(f"--- DEBUG: Sensitivity={sensitivity}, Scale={scale}, Epsilon={budget} ---")

        dp_dfg = {edge: max(0, count + int(self.rng.laplace(loc=0.0, scale=scale, size=1)[0])) for edge, count in
                  self.dfr.items()}

        return dp_dfg

    def get_unfiltered_start_activities(self):
        return {activity: count for (start, activity), count in self.dfr.items() if start == "START" and count > 0}

    def get_end_activities(self):
        return {activity: count for (activity, end), count in self.dfr.items() if end == "END" and count > 0}

