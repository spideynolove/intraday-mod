from __future__ import annotations
from collections import namedtuple, OrderedDict
from typing import Optional, Sequence
import numpy as np
import gymnasium as gym
from ..feature import Feature
from ..frame import Frame

_Output = namedtuple("TickMicrostructureOutput", [
    "buy_tick_ratio",
    "vwap_deviation",
    "spread_mean_norm",
    "spread_expansion",
    "flip_rate",
])


class TickMicrostructure(Feature):
    def __init__(self, write_to: str = "state", **kwargs):
        super().__init__(write_to=write_to, **kwargs)
        self.names = list(_Output._fields)
        self.spaces = OrderedDict(
            (name, gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32))
            for name in self.names
        )

    def process(self, frames: Sequence[Frame], state: OrderedDict) -> Optional[OrderedDict]:
        frame = frames[-1]
        ticks = frame.ticks or 1
        buy_ratio = (frame.buy_ticks or 0) / ticks
        vwap = frame.vwap or frame.close
        vwap_dev = (frame.close - vwap) / vwap if vwap else 0.0
        avg_spread = frame.avg_trade_spread or 1e-8
        spread_norm = avg_spread / frame.close if frame.close else 0.0
        spread_exp = (frame.trade_spread_max or avg_spread) / avg_spread
        flip_rate = (frame.flips or 0) / ticks
        result = _Output(
            buy_tick_ratio=float(buy_ratio),
            vwap_deviation=float(vwap_dev),
            spread_mean_norm=float(spread_norm),
            spread_expansion=float(spread_exp),
            flip_rate=float(flip_rate),
        )
        if self.write_to_state:
            state["tick_microstructure"] = result
        return state
