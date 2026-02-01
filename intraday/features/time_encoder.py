from typing import Sequence, Union, Literal
from collections import OrderedDict
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class TimeEncoder(Feature):
    def __init__(
        self,
        source: Union[str, Sequence[str]] = "time_start",
        yday=True,
        wday=True,
        write_to: Literal["frame", "state", "both"] = "state",
    ):
        super().__init__(write_to=write_to)
        if isinstance(source, str):
            self.source = [source]
        elif isinstance(source, Sequence):
            self.source = source
        else:
            raise ValueError
        self.yday = yday
        self.wday = wday
        for name in self.source:
            assert isinstance(name, str)
            self.names.append(f"yday_{name}")
            self.names.append(f"wday_{name}")
            self.names.append(f"time_{name}")
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict(
                {name: gym.spaces.Box(0, 1, shape=(1,)) for name in self.names}
            )
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        last_frame = frames[-1]
        for i, name in enumerate(self.source):
            datetime = getattr(last_frame, name)
            year_start = datetime.replace(
                month=1, day=1, hour=0, minute=0, second=0, microsecond=0
            )
            year_end = datetime.replace(
                month=12,
                day=31,
                hour=23,
                minute=59,
                second=59,
                microsecond=999999,
            )
            day_start = datetime.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            if self.yday:
                yday = (day_start - year_start) / (year_end - year_start)
            else:
                yday = 0.0
            if self.wday:
                wday = datetime.isoweekday() / 7
            else:
                wday = 0.0
            time = (datetime.timestamp() - day_start.timestamp()) / float(
                24 * 60 * 60
            )
            if self.write_to_frame:
                setattr(last_frame, self.names[i * 3 + 0], yday)
                setattr(last_frame, self.names[i * 3 + 1], wday)
                setattr(last_frame, self.names[i * 3 + 2], time)
            if self.write_to_state:
                state[self.names[i * 3 + 0]] = yday
                state[self.names[i * 3 + 1]] = wday
                state[self.names[i * 3 + 2]] = time

    def __repr__(self):
        return f"{self.__class__.__name__}(source={self.source}, write_to={self.write_to})"
