from typing import Sequence, Tuple, Union, Literal
from collections import OrderedDict, namedtuple
from numbers import Real
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature
from intraday.features.gaussian_smooth import GaussianSmooth

Extremum = namedtuple("Extremum", "frame sign level", defaults=(None, 0, None))


class Fractal(Feature):
    def __init__(
        self,
        radius: int,
        source: Union[str, Tuple[str, str]] = ("low", "high"),
        threshold: Union[Real, str, None] = None,
        level: Union[Real, Sequence[Real]] = 1.0,
        write_to: Literal["frame", "state", "both"] = "both",
        limit: int = 1000,
    ):
        super().__init__(write_to=write_to, period=2 * radius + 1)
        assert isinstance(radius, int) and radius > 0
        self.radius = radius
        if isinstance(source, str):
            self.field = f"fractal_{radius}_{source}"
        elif isinstance(source, Tuple) and len(source) == 2:
            self.field = f"fractal_{radius}_{source[0]}_{source[1]}"
        else:
            raise ValueError("Invalid source")
        self.source = source
        assert (
            threshold is None
            or isinstance(threshold, str)
            or isinstance(threshold, Real)
            and threshold > 0
        )
        self.threshold = threshold
        if isinstance(level, Real):
            self.level = [level]
        elif isinstance(level, Sequence):
            self.level = level
        else:
            raise ValueError("Invalid level")
        if self.threshold is not None:
            for level in self.level:
                level_str = ("%.3g" % level).replace(".", "_")
                self.names.append(f"fractal_support_threshold_{level_str}")
                self.names.append(f"fractal_resistance_threshold_{level_str}")
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict(
                {
                    name: gym.spaces.Box(-math.inf, math.inf, shape=(1,))
                    for name in self.names
                }
            )
        else:
            self.spaces = OrderedDict()
        assert isinstance(limit, int) and limit > 0
        self.limit = limit
        self.extremums = []

    def reset(self):
        self.extremums.clear()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        R = self.radius
        N = len(frames)
        last_frame = frames[-1]
        setattr(last_frame, self.field, 0)
        i = N - R - 1
        if i >= 0:
            central_frame = frames[i]
            if isinstance(self.source, str):
                source_low = source_high = self.source
            else:
                source_low, source_high = self.source
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
            sign = 0
            if He >= highest:
                discard = False
                for j in range(len(self.extremums) - 1, -1, -1):
                    e = self.extremums[j]
                    if e.sign < 0:
                        if e.level > highest:
                            discard = True
                            break
                        else:
                            setattr(self.extremums[j].frame, self.field, 0)
                            del self.extremums[j]
                    elif e.sign > 0:
                        break
                if not discard:
                    self.extremums.append(
                        Extremum(frame=central_frame, sign=-1, level=highest)
                    )
                    sign = -1
            elif Le <= lowest:
                discard = False
                for j in range(len(self.extremums) - 1, -1, -1):
                    e = self.extremums[j]
                    if e.sign > 0:
                        if e.level < lowest:
                            discard = True
                            break
                        else:
                            setattr(self.extremums[j].frame, self.field, 0)
                            del self.extremums[j]
                    elif e.sign < 0:
                        break
                if not discard:
                    self.extremums.append(
                        Extremum(frame=central_frame, sign=1, level=lowest)
                    )
                    sign = 1
            if len(self.extremums) > self.limit:
                del self.extremums[0]
            setattr(central_frame, self.field, sign)
        if self.threshold is not None:
            if isinstance(self.threshold, str):
                threshold = getattr(last_frame, self.threshold)
            else:
                threshold = self.threshold
            for i, level in enumerate(self.level):
                support, resistance = self._support_resistance(
                    price=last_frame.close,
                    extremums=self.extremums,
                    threshold=threshold * level,
                )
                if self.write_to_frame:
                    setattr(last_frame, self.names[0 + i * 2], support)
                    setattr(last_frame, self.names[1 + i * 2], resistance)
                if self.write_to_state:
                    state[self.names[0 + i * 2]] = support
                    state[self.names[1 + i * 2]] = resistance

    @staticmethod
    def _support_resistance(
        price: Real, extremums: Sequence[Extremum], threshold: Real
    ):
        support, resistance = 0, 0
        for extremum in extremums:
            if price < extremum.level <= price + threshold:
                resistance += 1
            elif price > extremum.level >= price - threshold:
                support += 1
        return support, resistance

    @staticmethod
    def _support_resistance_window(
        price: Real,
        extremums: Sequence[Extremum],
        window_radius: int = 10,
        level_radius: int = 4,
        price_step: Real = 1,
    ):
        kernel = GaussianSmooth._get_gaussian_kernel(radius=level_radius)
        c = math.floor(0.5 + price / price_step)
        window = {
            ((c + i) * price_step): (0.0)
            for i in range(-window_radius, window_radius + 1)
        }
        for extremum in extremums:
            e = math.floor(0.5 + extremum.level / price_step)
            if abs(c - e) <= window_radius + level_radius:
                for k in range(2 * level_radius + 1):
                    ei = e + k - level_radius
                    if abs(c - ei) <= window_radius:
                        window[ei * price_step] += kernel[k]
        return window

    def __repr__(self):
        return f"{self.__class__.__name__}(radius={self.radius}, source={self.source},threshold={self.threshold}, level={self.level}, write_to={self.write_to})"
