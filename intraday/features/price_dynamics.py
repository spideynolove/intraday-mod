from typing import Sequence, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.processor import Trade
from intraday.feature import TradesFeature


class PriceDynamics(TradesFeature):
    def __init__(self, write_to: Literal["frame", "state", "both"] = "state"):
        super().__init__(write_to=write_to)
        self.names = [
            "buy_price_move_up",
            "buy_volume_move_up",
            "buy_move_up_ease",
            "buy_price_move_down",
            "buy_volume_move_down",
            "buy_trades_price_still",
            "buy_volume_price_still",
            "sell_price_move_up",
            "sell_volume_move_up",
            "sell_price_move_down",
            "sell_volume_move_down",
            "sell_move_down_ease",
            "sell_trades_price_still",
            "sell_volume_price_still",
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
        self.last_price = None
        self.last_buy_price = None
        self.last_sell_price = None
        self.buy_price_move_up = 0
        self.buy_volume_move_up = 0
        self.buy_price_move_down = 0
        self.buy_volume_move_down = 0
        self.buy_trades_price_still = 0
        self.buy_volume_price_still = 0
        self.sell_price_move_up = 0
        self.sell_volume_move_up = 0
        self.sell_price_move_down = 0
        self.sell_volume_move_down = 0
        self.sell_trades_price_still = 0
        self.sell_volume_price_still = 0

    def reset(self):
        self.last_price = None
        self.last_buy_price = None
        self.last_sell_price = None
        self.buy_price_move_up = 0
        self.buy_volume_move_up = 0
        self.buy_price_move_down = 0
        self.buy_volume_move_down = 0
        self.buy_trades_price_still = 0
        self.buy_volume_price_still = 0
        self.sell_price_move_up = 0
        self.sell_volume_move_up = 0
        self.sell_price_move_down = 0
        self.sell_volume_move_down = 0
        self.sell_trades_price_still = 0
        self.sell_volume_price_still = 0

    def update(self, trades: Sequence[Trade]):
        trade = trades[-1]
        if trade.operation == "B":
            if self.last_price is not None and self.last_price == trade.price:
                self.buy_trades_price_still += 1
                self.buy_volume_price_still += trade.amount
            if self.last_buy_price is not None:
                if self.last_buy_price <= trade.price:
                    self.buy_price_move_up += trade.price - self.last_buy_price
                    self.buy_volume_move_up += trade.amount
                else:
                    self.buy_price_move_down += (
                        self.last_buy_price - trade.price
                    )
                    self.buy_volume_move_down += trade.amount
            self.last_buy_price = trade.price
        else:
            if self.last_price is not None and self.last_price == trade.price:
                self.sell_trades_price_still += 1
                self.sell_volume_price_still += trade.amount
            if self.last_sell_price is not None:
                if self.last_sell_price >= trade.price:
                    self.sell_price_move_down += (
                        self.last_sell_price - trade.price
                    )
                    self.sell_volume_move_down += trade.amount
                else:
                    self.sell_price_move_up += (
                        trade.price - self.last_sell_price
                    )
                    self.sell_volume_move_up += trade.amount
            self.last_sell_price = trade.price
        self.last_price = trade.price

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        buy_move_up_ease = (
            self.buy_price_move_up / self.buy_volume_move_up
            if self.buy_volume_move_up != 0
            else 0.0
        )
        sell_move_down_ease = (
            self.sell_price_move_down / self.sell_volume_move_down
            if self.sell_volume_move_down != 0
            else 0.0
        )
        if self.write_to_frame:
            setattr(frame, "buy_price_move_up", self.buy_price_move_up)
            setattr(frame, "buy_volume_move_up", self.buy_volume_move_up)
            setattr(frame, "buy_move_up_ease", buy_move_up_ease)
            setattr(frame, "buy_price_move_down", self.buy_price_move_down)
            setattr(frame, "buy_volume_move_down", self.buy_volume_move_down)
            setattr(
                frame, "buy_trades_price_still", self.buy_trades_price_still
            )
            setattr(
                frame, "buy_volume_price_still", self.buy_volume_price_still
            )
            setattr(frame, "sell_price_move_down", self.sell_price_move_down)
            setattr(frame, "sell_volume_move_down", self.sell_volume_move_down)
            setattr(frame, "sell_move_down_ease", sell_move_down_ease)
            setattr(frame, "sell_price_move_up", self.sell_price_move_up)
            setattr(frame, "sell_volume_move_up", self.sell_volume_move_up)
            setattr(
                frame, "sell_trades_price_still", self.sell_trades_price_still
            )
            setattr(
                frame, "sell_volume_price_still", self.sell_volume_price_still
            )
        if self.write_to_state:
            state["buy_price_move_up"] = self.buy_price_move_up
            state["buy_volume_move_up"] = self.buy_volume_move_up
            state["buy_move_up_ease"] = buy_move_up_ease
            state["buy_price_move_down"] = self.buy_price_move_down
            state["buy_volume_move_down"] = self.buy_volume_move_down
            state["buy_trades_price_still"] = self.buy_trades_price_still
            state["buy_volume_price_still"] = self.buy_volume_price_still
            state["sell_price_move_down"] = self.sell_price_move_down
            state["sell_volume_move_down"] = self.sell_volume_move_down
            state["sell_move_down_ease"] = sell_move_down_ease
            state["sell_price_move_up"] = self.sell_price_move_up
            state["sell_volume_move_up"] = self.sell_volume_move_up
            state["sell_trades_price_still"] = self.sell_trades_price_still
            state["sell_volume_price_still"] = self.sell_volume_price_still
        self.buy_price_move_up = 0
        self.buy_volume_move_up = 0
        self.buy_price_move_down = 0
        self.buy_volume_move_down = 0
        self.buy_trades_price_still = 0
        self.buy_volume_price_still = 0
        self.sell_price_move_up = 0
        self.sell_volume_move_up = 0
        self.sell_price_move_down = 0
        self.sell_volume_move_down = 0
        self.sell_trades_price_still = 0
        self.sell_volume_price_still = 0

    def __repr__(self):
        return f"{self.__class__.__name__}(write_to={self.write_to})"
