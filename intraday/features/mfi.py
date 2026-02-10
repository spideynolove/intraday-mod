from typing import Sequence, Literal
from collections import OrderedDict
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class MFI(Feature):
    def __init__(
        self,
        period: int = 14,
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=period)
        self.names = [f'mfi_{period}']
        self._prev_tp = None

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(0, 100, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def reset(self):
        self._prev_tp = None

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        window_size = min(len(frames), self.period + 1)
        window = frames[-window_size:]

        positive_flow = 0.0
        negative_flow = 0.0

        for i in range(1, len(window)):
            tp = (window[i].high + window[i].low + window[i].close) / 3
            prev_tp = (window[i-1].high + window[i-1].low + window[i-1].close) / 3
            raw_money_flow = tp * window[i].volume

            if tp > prev_tp:
                positive_flow += raw_money_flow
            elif tp < prev_tp:
                negative_flow += raw_money_flow

        if negative_flow == 0:
            mfi = 100.0
        else:
            money_ratio = positive_flow / negative_flow
            mfi = 100 - (100 / (1 + money_ratio))

        if self.write_to_frame:
            setattr(frames[-1], self.names[0], mfi)
        if self.write_to_state:
            state[self.names[0]] = mfi
