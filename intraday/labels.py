import math
import numpy as np
from typing import Sequence, MutableSequence, Tuple, Any, Union
from collections import OrderedDict, namedtuple
from numbers import Real
from .frame import Frame

Event = namedtuple("Event", "frame sign level", defaults=(None, 0, None))


class Labels:
    @staticmethod
    def apply_gaussian_filter(
        frames: Sequence[Frame], radius: int, source: str = "vwap"
    ) -> str:
        kernel = Labels._get_gaussian_kernel(radius)
        N = len(frames)
        R = radius
        P = 2 * R + 1
        dest = f"gauss_{radius}_{source}"
        for i, frame in enumerate(frames):
            value = 0.0
            for k in range(P):
                if i - R + k < 0:
                    v = getattr(frames[0], source)
                elif i - R + k >= N:
                    v = getattr(frames[-1], source)
                else:
                    v = getattr(frames[i - R + k], source)
                value += kernel[k] * v
            setattr(frame, dest, value)
        return dest

    _gaussian_kernels = {}

    @staticmethod
    def _get_gaussian_kernel(radius: int):
        assert isinstance(radius, int) and radius > 0
        if radius in Labels._gaussian_kernels:
            return Labels._gaussian_kernels[radius]
        sigma = radius / 2
        norm = 1.0 / (math.sqrt(2 * math.pi) * sigma)
        coeff = 2 * sigma * sigma
        total = 0.0
        kernel = []
        for x in range(-radius, radius + 1):
            g = norm * math.exp(-x * x / coeff)
            kernel.append(g)
            total += g
        for i in range(len(kernel)):
            kernel[i] = kernel[i] / total
        kernel = tuple(kernel)
        Labels._gaussian_kernels[radius] = kernel
        return kernel

    @staticmethod
    def calculate_standard_deviation(
        frames: Sequence[Frame],
        expected: str,
        variable: Union[str, Sequence[str]] = ("low", "high", "close"),
    ) -> float:
        dy = []
        for frame in frames:
            expected_value = getattr(frame, expected)
            if isinstance(variable, str):
                variable_value = getattr(frame, variable)
                dy.append(variable_value - expected_value)
            else:
                for field in variable:
                    variable_value = getattr(frame, field)
                    dy.append(variable_value - expected_value)
        dy = np.array(dy)
        return np.sqrt(np.mean(dy**2))

    @staticmethod
    def _calculate_linear_regression(
        x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, float]:
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        dx = x - x_mean
        dy = y - y_mean
        k = np.dot(dx, dy) / np.dot(dx, dx)
        b = y_mean - k * x_mean
        return k, b

    @staticmethod
    def calculate_linear_regression(
        frames: Sequence[Frame], x_name: str, y_name: str
    ) -> Tuple[float, float]:
        x, y = [], []
        for frame in frames:
            x.append(getattr(frame, x_name))
            y.append(getattr(frame, y_name))
        x, y = np.array(x), np.array(y)
        return Labels._calculate_linear_regression(x, y)

    @staticmethod
    def fractal_extremums_filter(
        frames: Sequence[Frame],
        radius: int = 2,
        source: Union[str, Tuple[str, str]] = ("low", "high"),
    ) -> MutableSequence[Event]:
        assert isinstance(radius, int) and radius > 0
        R = radius
        N = len(frames)
        if isinstance(source, str):
            source_low = source_high = source
        elif isinstance(source, Tuple):
            source_low, source_high = source
        else:
            raise ValueError("Invalid source")
        extremums = []
        for i, central_frame in enumerate(frames):
            highest, lowest = None, None
            for frame in frames[max(0, i - R) : min(N - 1, i + R + 1)]:
                low = getattr(frame, source_low)
                high = getattr(frame, source_high)
                if lowest is None or lowest > low:
                    lowest = low
                if highest is None or highest < high:
                    highest = high
            Le = getattr(central_frame, source_low)
            He = getattr(central_frame, source_high)
            if He >= highest:
                discard = False
                for j in range(len(extremums) - 1, -1, -1):
                    e = extremums[j]
                    if e.sign < 0:
                        if e.level > highest:
                            discard = True
                            break
                        else:
                            del extremums[j]
                    elif e.sign > 0:
                        break
                if not discard:
                    extremums.append(
                        Event(frame=central_frame, sign=-1, level=highest)
                    )
            elif Le <= lowest:
                discard = False
                for j in range(len(extremums) - 1, -1, -1):
                    e = extremums[j]
                    if e.sign > 0:
                        if e.level < lowest:
                            discard = True
                            break
                        else:
                            del extremums[j]
                    elif e.sign < 0:
                        break
                if not discard:
                    extremums.append(
                        Event(frame=central_frame, sign=1, level=lowest)
                    )
            else:
                pass
        return extremums

    @staticmethod
    def cumulative_sum_filter(
        frames: Sequence[Frame],
        threshold: Real,
        source: str = "close",
        prediction: Union[str, None] = None,
    ) -> MutableSequence[Event]:
        assert isinstance(threshold, Real) and threshold > 0.0
        expected = getattr(frames[0], source)
        pos, neg = 0.0, 0.0
        events = []
        for i, frame in enumerate(frames):
            value = getattr(frame, source)
            pos = max(0, pos + value - expected)
            neg = min(0, neg + value - expected)
            if pos >= threshold:
                events.append(Event(frame=frame, sign=1, level=value))
                pos = 0
            if neg <= -threshold:
                events.append(Event(frame=frame, sign=-1, level=value))
                neg = 0
            expected = (
                getattr(frame, prediction) if prediction is not None else value
            )
        return events

    @staticmethod
    def apply_triple_barrier(
        frames: Sequence[Frame],
        events: Sequence[Event],
        barrier: Union[float, Tuple[float, float]],
    ) -> MutableSequence[Event]:
        lower_k, upper_k = (
            barrier if isinstance(barrier, Tuple) else (barrier, barrier)
        )
        out = []
        for j, event in enumerate(events):
            event_frame = event.frame
            i0 = next((i for i, f in enumerate(frames) if f is event_frame), 0)
            i1 = (
                next((i for i, f in enumerate(frames) if f is events[j + 1].frame), len(frames))
                if j + 1 < len(events)
                else len(frames)
            )
            base_price = event_frame.close
            upper_barrier = base_price + upper_k
            lower_barrier = base_price - lower_k
            upper_hit_i, lower_hit_i = None, None
            for i in range(i0, i1):
                frame = frames[i]
                if upper_hit_i is None and frame.high > upper_barrier:
                    upper_hit_i = i
                if lower_hit_i is None and frame.low < lower_barrier:
                    lower_hit_i = i
                if upper_hit_i is not None and lower_hit_i is not None:
                    break
