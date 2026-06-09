import threading

from pm4py.streaming.stream.live_event_stream import LiveEventStream, StreamState


class PausableLiveEventStream(LiveEventStream):

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.is_paused = False
        self._futures = set()

        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)



    def _track_future(self, future):
        self._futures.add(future)

        def _done_callback(fut): # need to use RLock here, as this method can be executed in place in the same (already blocked thread)
            with self._cond:
                self._futures.discard(fut)
                self._cond.notify()

        future.add_done_callback(_done_callback)

    def _deliver(self):
        while self._state != StreamState.INACTIVE:
            self._cond.acquire()
            while len(self._dq) == 0:
                self._cond.notify()
                if self._state != StreamState.FINISHED:
                    self._cond.wait()
                else:
                    self._cond.release()
                    return
            event = self._dq.popleft()
            for algo in self._observers:
                future = self._tp.submit(algo.receive, event)
                self._track_future(future)

            self._cond.release()

    def append(self, event):
        """
        Appends a new event to the end of the queue.
        If the stream is paused, the appending is postponed until resume() is called.
        :param event: event to append
        :return:
        """
        self._cond.acquire()


        # Blocks the appending of new events, while stream is paused

        while self.is_paused:

            self._cond.notify()
            self._cond.wait()

        if self._state != StreamState.FINISHED:
            self._dq.append(event)
            self._cond.notify()

        self._cond.release()

    def stop(self):
        self.pause()
        self._cond.acquire()
        while len(self._dq) > 0:
            self._cond.wait()
        self._tp.shutdown(wait=False, cancel_futures=True)
        if self._state == StreamState.ACTIVE:
            self._state = StreamState.FINISHED
            self._cond.notify()
        self._cond.release()


    def pause(self):
        """
        Pauses the appending of new events and waits until queue is empty.
        :return:
        """
        self._cond.acquire() # acquire lock
        self.is_paused = True

        while (len(self._dq)> 0 or len(self._futures) > 0) and self.is_paused and self.state != StreamState.FINISHED:
            # self._cond.notify()
            self._cond.wait()


        self._cond.release()

    def resume(self):
        """
        Resumes the appending of new events.
        :return:
        """
        self._cond.acquire()
        self.is_paused = False
        self._cond.notify()
        self._cond.release()

    def flush(self):
        """
        Combines pause() and resume():
        Stops the appending of new events and waits until queue is empty.
        Afterward, the stream is resumed immediately.
        :return:
        """
        self.pause()
        self.resume()