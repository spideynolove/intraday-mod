from typing import Sequence, Union, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


def price_raw(p0, p1):
    return p0


def price_delta(p0, p1):
    return p0 - p1 if p0 is not None and p1 is not None else 0.0


def price_return(p0, p1):
    return (
        p0 / p1 - 1.0 if p0 is not None and p1 is not None and p1 != 0 else 0.0
    )


def price_logreturn(p0, p1):
    assert p0 is None or p0 >= 0
    assert p1 is None or p1 >= 0
    return (
        math.log(p0 / p1)
        if p0 is not None and p1 is not None and p1 != 0
        else 0.0
    )


class PriceEncoder(Feature):
    Methods = {
        "raw": price_raw,
        "delta": price_delta,
        "return": price_return,
        "logreturn": price_logreturn,
    }

    def __init__(
        self,
        source: Union[str, Sequence[str]] = ("open", "high", "low", "close"),
        method: str = "delta",
        base: (None, str) = None,
        period: int = 2,
        write_to: Literal["frame", "state", "both"] = "state",
    ):
        assert isinstance(period, int) and period > 0
        super().__init__(period=period, write_to=write_to)
        assert isinstance(method, str) and method in PriceEncoder.Methods
        self.method = method
        assert base is None or isinstance(base, str)
        self.base = base
        if isinstance(source, str):
            self.source = [source]
        elif isinstance(source, Sequence):
            self.source = source
        else:
            raise ValueError
        for name in self.source:
            assert isinstance(name, str)
            if base is None:
                self.names.append(f"{method}_{period}_{name}")
            else:
                self.names.append(f"{method}_{period}_{name}_{base}")
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict(
                {
                    name: gym.spaces.Box(-math.inf, math.inf, shape=(1,))
                    for name in self.names
                }
            )
        else:
            self.spaces = OrderedDict()
        self._encode_price = PriceEncoder.Methods[method]

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        prev_frame = frames[-min(self.period, len(frames))]
        last_frame = frames[-1]
        price1 = None
        if self.base is not None:
            price1 = (
                getattr(prev_frame, self.base)
                if prev_frame is not None
                else None
            )
        for i, name in enumerate(self.source):
            price0 = getattr(last_frame, name)
            if self.base is None:
                price1 = (
                    getattr(prev_frame, name)
                    if prev_frame is not None
                    else None
                )
            price = self._encode_price(price0, price1)
            if self.write_to_frame:
                setattr(last_frame, self.names[i], price)
            if self.write_to_state:
                state[self.names[i]] = price

    def __repr__(self):
        return f"{self.__class__.__name__}(source={self.source}, method={self.method}, base={self.base},period={self.period}, write_to={self.write_to})"
