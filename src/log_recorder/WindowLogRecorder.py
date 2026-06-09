from collections import deque

import pm4py.objects.log.obj
from pm4py.streaming.algo.interface import StreamingAlgorithm
from pm4py.util import exec_utils
import pandas as pd


class WindowLogRecorder(StreamingAlgorithm):

    def __init__(self, parameters = None):

        if parameters is None:
            parameters = {}
            
        self.max_size = exec_utils.get_param_value("window_size", parameters, 50)
        
        self.window = deque()

        super().__init__(parameters)



    def _process(self, event):
        
        if len(self.window) < self.max_size:
            self.window.append(event)
        else:
            self.window.popleft()
            self.window.append(event)
            
    def _current_result(self):
        print(len(self.window))
        if len(self.window) == 0:
            return pm4py.EventLog()
        else:
            log =  pm4py.convert_to_event_log(pd.DataFrame(self.window))
            print(sum([len(t) for t in log]))
            return log
        
        
def apply(window_size = int, parameters: dict = None):
    
    if parameters is None:
        parameters = {}
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    
    parameters["window_size"] = window_size
    
    return WindowLogRecorder(parameters)