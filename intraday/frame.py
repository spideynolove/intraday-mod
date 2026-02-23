from __future__ import annotations
from numbers import Real
from typing import Sequence, Union, Optional
from arrow import Arrow
from datetime import timedelta
from .provider import Trade, TradeOI


class Frame(object):
    def __init__(self, **kwargs):
        self.time_start: Optional[Arrow] = None

        self.time_end: Optional[Arrow] = None

        self.duration: Optional[timedelta] = None

        self.prev_close: Optional[Real] = None

        self.open: Optional[Real] = None

        self.high: Optional[Real] = None

        self.low: Optional[Real] = None

        self.close: Optional[Real] = None

        self.hl2: Optional[Real] = None

        self.hlc3: Optional[Real] = None

        self.true_range: Optional[Real] = 0

        self.flips: int = 0

        self._sum_spread: Real = 0

        self.avg_trade_spread: Optional[Real] = None

        self.trade_spread_min: Optional[Real] = None

        self.trade_spread_max: Optional[Real] = None

        self.ticks: int = 0

        self.volume: Real = 0

        self.money: Real = 0

        self.vwap: Optional[Real] = None

        self.avg_trade_tick: Optional[float] = None

        self.avg_trade_spread: Optional[Real] = None

        self.avg_trade_amount: Optional[Real] = None

        self.buy_ticks: int = 0

        self.buy_volume: Real = 0

        self.buy_money: Real = 0

        self.buy_vwap: Optional[Real] = None

        self.avg_buy_amount: Optional[Real] = None

        self.avg_buy_money: Optional[Real] = None

        self.sell_ticks: int = 0

        self.sell_volume: Real = 0

        self.sell_money: Real = 0

        self.sell_vwap: Optional[Real] = None

        self.avg_sell_amount: Optional[Real] = None

        self.avg_sell_money = None

        self.conseq_buy_ticks: int = 0

        self.conseq_buy_volume: Real = 0

        self.conseq_buy_money: Real = 0

        self.conseq_sell_ticks: Real = 0

        self.conseq_sell_volume: Real = 0

        self.conseq_sell_money: Real = 0

        self.vwap_range: Real = 0

        self.imbalance_ticks: int = 0

        self.imbalance_volume: Real = 0

        self.imbalance_money: Real = 0

        self.imbalance_conseq_ticks: int = 0

        self.imbalance_conseq_volume: Real = 0

        self.imbalance_conseq_money: Real = 0

        self.oi_open: Optional[Real] = None
        self.oi_open: Optional[Real] = None

        self.oi_high: Optional[Real] = None

        self.oi_low: Optional[Real] = None

        self.oi_close: Optional[Real] = None

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__dict__ != other.__dict__

    def update(self, trades: Sequence[Union[Trade, TradeOI]]):
        trade = trades[-1]
        prev_trade = trades[-2] if len(trades) > 1 else None
        datetime, operation, amount, price = (
            trade.datetime,
            trade.operation,
            trade.amount,
            trade.price,
        )
        money = amount * price
        if self.time_start is None or self.time_start > datetime:
            self.time_start = datetime
        if self.time_end is None or self.time_end < datetime:
            self.time_end = datetime
        self.duration = (self.time_end - self.time_start).total_seconds()
        if self.prev_close is None and prev_trade is not None:
            self.prev_close = prev_trade.price
        if self.open is None:
            self.open = price
        if self.high is None or self.high < price:
            self.high = price
        if self.low is None or self.low > price:
            self.low = price
        self.close = price
        if prev_trade is not None and prev_trade.operation != trade.operation:
            spread = abs(prev_trade.price - trade.price)
            self._sum_spread += spread
            self.flips += 1
            self.avg_trade_spread = self._sum_spread / self.flips
            if self.trade_spread_min is None or self.trade_spread_min > spread:
                self.trade_spread_min = spread
            if self.trade_spread_max is None or self.trade_spread_max < spread:
                self.trade_spread_max = spread
        if hasattr(trade, "open_interest"):
            open_interest = trade.open_interest
            if self.oi_open is None:
                self.oi_open = open_interest
            if self.oi_high is None or self.oi_high < open_interest:
                self.oi_high = open_interest
            if self.oi_low is None or self.oi_low > open_interest:
                self.oi_low = open_interest
            self.oi_close = open_interest
        self.ticks += 1
        self.volume += amount
        self.money += money
        if operation == "B":
            self.buy_ticks += 1
            self.buy_volume += amount
            self.buy_money += money
        else:
            self.sell_ticks += 1
            self.sell_volume += amount
            self.sell_money += money
        if prev_trade is not None:
            if prev_trade.operation == operation:
                if operation == "B":
                    self.conseq_buy_ticks += 1
                    self.conseq_buy_volume += amount
                    self.conseq_buy_money += money
                else:
                    self.conseq_sell_ticks += 1
                    self.conseq_sell_volume += amount
                    self.conseq_sell_money += money

    def finalize(self) -> Frame:
        self.hl2 = (self.high + self.low) / 2
        self.hlc3 = (self.high + self.low + self.close) / 3
        if self.prev_close is not None:
            self.true_range = max(self.high, self.prev_close) - min(
                self.low, self.prev_close
            )
        else:
            self.true_range = self.high - self.low
        if self.ticks > 0:
            self.avg_trade_amount = self.volume / self.ticks
        if self.buy_ticks > 0:
            self.avg_buy_amount = self.buy_volume / self.buy_ticks
            self.avg_buy_money = self.buy_money / self.buy_ticks
        if self.sell_ticks > 0:
            self.avg_sell_amount = self.sell_volume / self.sell_ticks
            self.avg_sell_money = self.sell_money / self.sell_ticks
        if self.volume > 0:
            self.vwap = self.money / self.volume
        if self.buy_volume > 0:
            self.buy_vwap = self.buy_money / self.buy_volume
        if self.sell_volume > 0:
            self.sell_vwap = self.sell_money / self.sell_volume
        self.vwap_range = _diff(self.sell_vwap, self.buy_vwap)
        self.imbalance_ticks = self.buy_ticks - self.sell_ticks
        self.imbalance_volume = self.buy_volume - self.sell_volume
        self.imbalance_money = self.buy_money - self.sell_money
        self.imbalance_conseq_ticks = (
            self.conseq_buy_ticks - self.conseq_sell_ticks
        )
        self.imbalance_conseq_volume = (
            self.conseq_buy_volume - self.conseq_sell_volume
        )
        self.imbalance_conseq_money = (
            self.conseq_buy_money - self.conseq_sell_money
        )
        if self.ticks > 0:
            self.avg_trade_tick = self.imbalance_ticks / self.ticks
        return self

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            + ",".join(
                [f"{str(k)}={str(v)}" for k, v in self.__dict__.items()]
            )
            + ")"
        )


def _diff(v1, v2):
    return v1 - v2 if v1 is not None and v2 is not None else 0.0


try:
    from ._frame import Frame
except ImportError:
    pass
