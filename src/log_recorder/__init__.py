import pandas as pd
import pm4py
from pm4py.streaming.algo.interface import StreamingAlgorithm


class FullLogRecorder(StreamingAlgorithm):
    """Accumulates all events seen so far; returns a full pm4py EventLog."""

    def __init__(self, parameters=None):
        if parameters is None:
            parameters = {}
        self.events: list[dict] = []
        super().__init__(parameters)

    def _process(self, event):
        # store a plain dict copy (the original event object may be mutated)
        self.events.append(dict(event))

    def _current_result(self):
        if not self.events:
            return pm4py.objects.log.obj.EventLog()
        return pm4py.convert_to_event_log(pd.DataFrame(self.events))
