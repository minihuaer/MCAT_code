"""Bounded background hand detection with source-frame metadata.

No MediaPipe dependency: the caller supplies a detector returning
(full-image wrist pixel, landmarks, world landmarks, source crop).
"""
from time import perf_counter

import numpy as np


class AsyncHandObservation:
    """At most one job per recognizer; polling never waits for inference.

    The frame, crop centre and KDU pixel belong to the same source timestamp.
    Busy workers skip input frames instead of building an inference queue.
    """
    def __init__(self, pool, detect, recognizer):
        self.pool, self.detect, self.recognizer = pool, detect, recognizer
        self.pending = None
        self.source = None

    def poll(self, frame, center, timestamp_ms):
        completed = None
        if self.pending is not None and self.pending.done():
            pending, source = self.pending, self.source
            self.pending = None
            result, elapsed = pending.result()  # done; propagate worker failures
            completed = (*result, source[0], source[1], elapsed)
        if self.pending is None:
            center = np.asarray(center, dtype=float)[:2].copy()
            self.source = (int(timestamp_ms), center)
            # Capture this array by reference; the caller never modifies RGB
            # after submission. Only one frame per worker remains in memory.
            def run():
                started = perf_counter()
                result = self.detect(frame, center, self.recognizer, int(timestamp_ms))
                return result, (perf_counter() - started) * 1000.
            self.pending = self.pool.submit(run)
        return completed
