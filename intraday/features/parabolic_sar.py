from typing import Sequence, Tuple, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class ParabolicSAR(Feature):
    def __init__(
        self,
        acceleration: float = 0.02,
        max_velocity: float = 0.2,
        source: Tuple[str, str] = ("low", "high"),
        write_to: Literal["frame", "state", "both"] = "state",
    ):
        super().__init__(write_to=write_to)
        assert isinstance(acceleration, float) and 0 < acceleration < 1
        assert isinstance(max_velocity, float) and 0 < max_velocity < 1
        assert acceleration < max_velocity
        self.acceleration = acceleration
        self.max_velocity = max_velocity
        assert isinstance(source, Tuple) and len(source) == 2
        self.source = source
        a = f"{acceleration}"[2:]
        v = f"{max_velocity}"[2:]
        self.names = [
            f"psar_u_value_{a}_{v}_{source[0]}_{source[1]}",
            f"psar_u_reset_{a}_{v}_{source[0]}_{source[1]}",
            f"psar_d_value_{a}_{v}_{source[0]}_{source[1]}",
            f"psar_d_reset_{a}_{v}_{source[0]}_{source[1]}",
        ]
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict(
                {
                    name: gym.spaces.Box(-math.inf, math.inf, shape=(1,))
                    for name in self.names
                }
            )
        else:
            self.spaces = OrderedDict()
        self.u_velocity = 0
        self.d_velocity = 0
        self.lowest = None
        self.highest = None
        self.u_sar = None
        self.d_sar = None

    def reset(self):
        self.u_velocity = 0
        self.d_velocity = 0
        self.lowest = None
        self.highest = None
        self.u_sar = None
        self.d_sar = None

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        last_frame = frames[-1]
        low = getattr(last_frame, self.source[0])
        high = getattr(last_frame, self.source[1])
        if self.u_velocity == 0 or self.highest is None or self.highest < high:
            self.highest = high
            self.u_velocity = min(
                self.max_velocity, self.u_velocity + self.acceleration
            )
        prior = self.u_sar if self.u_sar is not None else low
        self.u_sar = prior + self.u_velocity * (self.highest - prior)
        u_reset = 0
        if self.u_sar >= low:
            self.u_velocity = 0
            self.u_sar = low
            self.highest = high
            u_reset = 1
        if len(frames) > 1:
            low1 = getattr(frames[-2], self.source[0])
            if self.u_sar > low1:
                self.u_sar = low1
        if len(frames) > 2:
            low2 = getattr(frames[-3], self.source[0])
            if self.u_sar > low2:
                self.u_sar = low2
        if self.d_velocity == 0 or self.lowest is None or self.lowest > low:
            self.lowest = low
            self.d_velocity = min(
                self.max_velocity, self.d_velocity + self.acceleration
            )
        prior = self.d_sar if self.d_sar is not None else high
        self.d_sar = prior - self.d_velocity * (prior - self.lowest)
        d_reset = 0
        if self.d_sar <= high:
            self.d_velocity = 0
            self.d_sar = high
            self.lowest = low
            d_reset = 1
        if len(frames) > 1:
            high1 = getattr(frames[-2], self.source[1])
            if self.d_sar < high1:
                self.d_sar = high1
        if len(frames) > 2:
            high2 = getattr(frames[-3], self.source[1])
            if self.d_sar < high2:
                self.d_sar = high2
        if self.write_to_frame:
            setattr(last_frame, self.names[0], self.u_sar)
            setattr(last_frame, self.names[1], u_reset)
            setattr(last_frame, self.names[2], self.d_sar)
            setattr(last_frame, self.names[3], d_reset)
        if self.write_to_state:
            state[self.names[0]] = self.u_sar
            state[self.names[1]] = u_reset
            state[self.names[2]] = self.d_sar
            state[self.names[3]] = d_reset
