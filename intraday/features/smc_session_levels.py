from typing import Sequence, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature

ASIAN_START, ASIAN_END = 0, 9
LONDON_START, LONDON_END = 7, 16
NY_START, NY_END = 12, 21
KILL_ZONES = set(range(7, 10)) | set(range(12, 15))


class SessionLevels(Feature):
    def __init__(self, write_to: Literal["state", "frame", "both"] = "state"):
        super().__init__(write_to=write_to)
        self.names = [
            'session_type',
            'in_kill_zone',
            'asian_high', 'asian_low', 'asian_range',
            'london_high', 'london_low', 'london_range',
            'ny_high', 'ny_low', 'ny_range',
            'prev_session_high', 'prev_session_low',
            'distance_to_session_high', 'distance_to_session_low',
        ]
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                'session_type': gym.spaces.Discrete(4),
                'in_kill_zone': gym.spaces.Discrete(2),
                'asian_high': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'asian_low': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'asian_range': gym.spaces.Box(0, math.inf, shape=(1,)),
                'london_high': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'london_low': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'london_range': gym.spaces.Box(0, math.inf, shape=(1,)),
                'ny_high': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'ny_low': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'ny_range': gym.spaces.Box(0, math.inf, shape=(1,)),
                'prev_session_high': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'prev_session_low': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'distance_to_session_high': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'distance_to_session_low': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
            })
        else:
            self.spaces = OrderedDict()
        self._asian_high = -math.inf
        self._asian_low = math.inf
        self._london_high = -math.inf
        self._london_low = math.inf
        self._ny_high = -math.inf
        self._ny_low = math.inf
        self._prev_high = 0.0
        self._prev_low = 0.0
        self._last_date = None
        self._last_session = 0

    def reset(self):
        self._asian_high = -math.inf
        self._asian_low = math.inf
        self._london_high = -math.inf
        self._london_low = math.inf
        self._ny_high = -math.inf
        self._ny_low = math.inf
        self._prev_high = 0.0
        self._prev_low = 0.0
        self._last_date = None
        self._last_session = 0

    def _session_type(self, hour: int) -> int:
        if NY_START <= hour < NY_END:
            return 3
        if LONDON_START <= hour < LONDON_END:
            return 2
        if ASIAN_START <= hour < ASIAN_END:
            return 1
        return 0

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        time = getattr(frame, 'time_end', None) or getattr(frame, 'time_start', None)

        if time is None:
            hour = 0
            date = None
        else:
            hour = time.hour
            date = time.date()

        if date and date != self._last_date:
            self._asian_high = -math.inf
            self._asian_low = math.inf
            self._london_high = -math.inf
            self._london_low = math.inf
            self._ny_high = -math.inf
            self._ny_low = math.inf
            self._last_date = date

        session = self._session_type(hour)

        if ASIAN_START <= hour < ASIAN_END:
            self._asian_high = max(self._asian_high, frame.high)
            self._asian_low = min(self._asian_low, frame.low)
        if LONDON_START <= hour < LONDON_END:
            self._london_high = max(self._london_high, frame.high)
            self._london_low = min(self._london_low, frame.low)
        if NY_START <= hour < NY_END:
            self._ny_high = max(self._ny_high, frame.high)
            self._ny_low = min(self._ny_low, frame.low)

        if session != self._last_session and self._last_session != 0:
            if self._last_session == 1:
                self._prev_high = self._asian_high if self._asian_high != -math.inf else 0.0
                self._prev_low = self._asian_low if self._asian_low != math.inf else 0.0
            elif self._last_session == 2:
                self._prev_high = self._london_high if self._london_high != -math.inf else 0.0
                self._prev_low = self._london_low if self._london_low != math.inf else 0.0
            elif self._last_session == 3:
                self._prev_high = self._ny_high if self._ny_high != -math.inf else 0.0
                self._prev_low = self._ny_low if self._ny_low != math.inf else 0.0
        self._last_session = session

        asian_h = self._asian_high if self._asian_high != -math.inf else 0.0
        asian_l = self._asian_low if self._asian_low != math.inf else 0.0
        london_h = self._london_high if self._london_high != -math.inf else 0.0
        london_l = self._london_low if self._london_low != math.inf else 0.0
        ny_h = self._ny_high if self._ny_high != -math.inf else 0.0
        ny_l = self._ny_low if self._ny_low != math.inf else 0.0

        session_high = {1: asian_h, 2: london_h, 3: ny_h}.get(session, 0.0)
        session_low = {1: asian_l, 2: london_l, 3: ny_l}.get(session, 0.0)

        values = [
            session, int(hour in KILL_ZONES),
            asian_h, asian_l, asian_h - asian_l,
            london_h, london_l, london_h - london_l,
            ny_h, ny_l, ny_h - ny_l,
            self._prev_high, self._prev_low,
            frame.close - session_high, frame.close - session_low,
        ]

        if self.write_to_frame:
            for name, val in zip(self.names, values):
                setattr(frame, name, val)
        if self.write_to_state:
            for name, val in zip(self.names, values):
                state[name] = val
