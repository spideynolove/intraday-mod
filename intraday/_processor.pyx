# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from ._frame cimport Frame


cdef class FastIntervalProcessor:

    cdef public str method
    cdef public int interval
    cdef public object duration
    cdef public object frame

    def __init__(self, method, int interval, duration=(1, 4 * 60 * 60), **kwargs):
        self.method = method
        self.interval = interval
        self.duration = duration
        self.frame = None

    @property
    def name(self):
        return f"{self.method}@{self.interval}"

    def reset(self):
        self.frame = None

    cpdef object process(self, trades):
        cdef Frame frame
        cdef object result, trade, time_start, time_end
        cdef double dur_min, dur_max

        result = None
        trade = trades[len(trades) - 1]
        dur_min = self.duration[0]
        dur_max = self.duration[1]

        if self.frame is None or (self.method == "time" and trade.datetime > self.frame.time_end):
            if self.frame is not None:
                result = (<Frame>self.frame).finalize()
            frame = Frame(time_start=trade.datetime)
            if self.method == "time":
                from .processor import _get_time_span
                time_start, time_end = _get_time_span(trade.datetime, self.interval)
                frame.time_start = time_start
                frame.time_end = time_end
            self.frame = frame
        else:
            frame = <Frame>self.frame

        frame.update(trades)

        if (
            frame.duration >= dur_min
            and (
                self.method == "tick" and frame.ticks >= self.interval
                or self.method == "volume" and frame.volume >= self.interval
                or self.method == "money" and frame.money >= self.interval
            )
            or frame.duration >= dur_max
        ):
            result = frame.finalize()
            self.frame = None

        return result

    def finish(self):
        if self.frame is None:
            return None
        result = (<Frame>self.frame).finalize()
        self.frame = None
        return result

    def __repr__(self):
        return f"FastIntervalProcessor(method={self.method}, interval={self.interval})"
