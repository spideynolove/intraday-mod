from typing import Sequence, Literal
from collections import OrderedDict
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class CMO(Feature):
    def __init__(
        self,
        period: int = 14,
        source: str = 'close',
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=period)
        self.source = source
        self.names = [f'cmo_{period}_{source}']

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(-100, 100, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        window_size = min(len(frames), self.period + 1)
        if window_size < 2:
            cmo = 0.0
        else:
            window = frames[-window_size:]
            gains = 0.0
            losses = 0.0

            for i in range(1, len(window)):
                change = getattr(window[i], self.source) - getattr(window[i-1], self.source)
                if change > 0:
                    gains += change
                else:
                    losses += abs(change)

            if gains + losses == 0:
                cmo = 0.0
            else:
                cmo = 100 * (gains - losses) / (gains + losses)

        if self.write_to_frame:
            setattr(frames[-1], self.names[0], cmo)
        if self.write_to_state:
            state[self.names[0]] = cmo
