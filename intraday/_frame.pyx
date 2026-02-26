# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from libc.math cimport isnan, fabs

cdef double _NaN = float('nan')


cdef class Frame:

    def __init__(self, **kwargs):
        self.time_start = None
        self.time_end = None
        self.duration = 0.0
        self._open = _NaN
        self._high = _NaN
        self._low = _NaN
        self._close = _NaN
        self._prev_close = _NaN
        self._hl2 = _NaN
        self._hlc3 = _NaN
        self._true_range = 0.0
        self.flips = 0
        self._sum_spread = 0.0
        self._avg_trade_spread = _NaN
        self._trade_spread_min = _NaN
        self._trade_spread_max = _NaN
        self.ticks = 0
        self.volume = 0.0
        self.money = 0.0
        self._vwap = _NaN
        self._avg_trade_tick = _NaN
        self._avg_trade_amount = _NaN
        self.buy_ticks = 0
        self.buy_volume = 0.0
        self.buy_money = 0.0
        self._buy_vwap = _NaN
        self._avg_buy_amount = _NaN
        self._avg_buy_money = _NaN
        self.sell_ticks = 0
        self.sell_volume = 0.0
        self.sell_money = 0.0
        self._sell_vwap = _NaN
        self._avg_sell_amount = _NaN
        self._avg_sell_money = _NaN
        self.conseq_buy_ticks = 0
        self.conseq_buy_volume = 0.0
        self.conseq_buy_money = 0.0
        self.conseq_sell_ticks = 0
        self.conseq_sell_volume = 0.0
        self.conseq_sell_money = 0.0
        self.vwap_range = 0.0
        self.imbalance_ticks = 0
        self.imbalance_volume = 0.0
        self.imbalance_money = 0.0
        self.imbalance_conseq_ticks = 0
        self.imbalance_conseq_volume = 0.0
        self.imbalance_conseq_money = 0.0
        self.oi_open = None
        self.oi_high = None
        self.oi_low = None
        self.oi_close = None
        for key, value in kwargs.items():
            setattr(self, key, value)

    # Properties mapping internal C doubles to Python-compatible Optional[float]
    @property
    def open(self): return None if isnan(self._open) else self._open
    @open.setter
    def open(self, v): self._open = _NaN if v is None else float(v)

    @property
    def high(self): return None if isnan(self._high) else self._high
    @high.setter
    def high(self, v): self._high = _NaN if v is None else float(v)

    @property
    def low(self): return None if isnan(self._low) else self._low
    @low.setter
    def low(self, v): self._low = _NaN if v is None else float(v)

    @property
    def close(self): return None if isnan(self._close) else self._close
    @close.setter
    def close(self, v): self._close = _NaN if v is None else float(v)

    @property
    def prev_close(self): return None if isnan(self._prev_close) else self._prev_close
    @prev_close.setter
    def prev_close(self, v): self._prev_close = _NaN if v is None else float(v)

    @property
    def hl2(self): return None if isnan(self._hl2) else self._hl2
    @hl2.setter
    def hl2(self, v): self._hl2 = _NaN if v is None else float(v)

    @property
    def hlc3(self): return None if isnan(self._hlc3) else self._hlc3
    @hlc3.setter
    def hlc3(self, v): self._hlc3 = _NaN if v is None else float(v)

    @property
    def true_range(self): return self._true_range
    @true_range.setter
    def true_range(self, v): self._true_range = 0.0 if v is None else float(v)

    @property
    def avg_trade_spread(self): return None if isnan(self._avg_trade_spread) else self._avg_trade_spread
    @avg_trade_spread.setter
    def avg_trade_spread(self, v): self._avg_trade_spread = _NaN if v is None else float(v)

    @property
    def trade_spread_min(self): return None if isnan(self._trade_spread_min) else self._trade_spread_min
    @trade_spread_min.setter
    def trade_spread_min(self, v): self._trade_spread_min = _NaN if v is None else float(v)

    @property
    def trade_spread_max(self): return None if isnan(self._trade_spread_max) else self._trade_spread_max
    @trade_spread_max.setter
    def trade_spread_max(self, v): self._trade_spread_max = _NaN if v is None else float(v)

    @property
    def vwap(self): return None if isnan(self._vwap) else self._vwap
    @vwap.setter
    def vwap(self, v): self._vwap = _NaN if v is None else float(v)

    @property
    def avg_trade_tick(self): return None if isnan(self._avg_trade_tick) else self._avg_trade_tick
    @avg_trade_tick.setter
    def avg_trade_tick(self, v): self._avg_trade_tick = _NaN if v is None else float(v)

    @property
    def avg_trade_amount(self): return None if isnan(self._avg_trade_amount) else self._avg_trade_amount
    @avg_trade_amount.setter
    def avg_trade_amount(self, v): self._avg_trade_amount = _NaN if v is None else float(v)

    @property
    def buy_vwap(self): return None if isnan(self._buy_vwap) else self._buy_vwap
    @buy_vwap.setter
    def buy_vwap(self, v): self._buy_vwap = _NaN if v is None else float(v)

    @property
    def avg_buy_amount(self): return None if isnan(self._avg_buy_amount) else self._avg_buy_amount
    @avg_buy_amount.setter
    def avg_buy_amount(self, v): self._avg_buy_amount = _NaN if v is None else float(v)

    @property
    def avg_buy_money(self): return None if isnan(self._avg_buy_money) else self._avg_buy_money
    @avg_buy_money.setter
    def avg_buy_money(self, v): self._avg_buy_money = _NaN if v is None else float(v)

    @property
    def sell_vwap(self): return None if isnan(self._sell_vwap) else self._sell_vwap
    @sell_vwap.setter
    def sell_vwap(self, v): self._sell_vwap = _NaN if v is None else float(v)

    @property
    def avg_sell_amount(self): return None if isnan(self._avg_sell_amount) else self._avg_sell_amount
    @avg_sell_amount.setter
    def avg_sell_amount(self, v): self._avg_sell_amount = _NaN if v is None else float(v)

    @property
    def avg_sell_money(self): return None if isnan(self._avg_sell_money) else self._avg_sell_money
    @avg_sell_money.setter
    def avg_sell_money(self, v): self._avg_sell_money = _NaN if v is None else float(v)

    cpdef update(self, trades):
        cdef double price, amount, money, spread
        cdef bint is_buy, same_op
        cdef object trade, prev_trade, datetime_obj

        cdef int _n = len(trades)
        trade = trades[_n - 1]
        prev_trade = trades[_n - 2] if _n > 1 else None
        price = trade.price
        amount = trade.amount
        is_buy = (trade.operation == "B")
        money = amount * price
        datetime_obj = trade.datetime

        if self.time_start is None or self.time_start > datetime_obj:
            self.time_start = datetime_obj
        if self.time_end is None or self.time_end < datetime_obj:
            self.time_end = datetime_obj
        self.duration = (self.time_end - self.time_start).total_seconds()

        if isnan(self._prev_close) and prev_trade is not None:
            self._prev_close = prev_trade.price
        if isnan(self._open):
            self._open = price
        if isnan(self._high) or self._high < price:
            self._high = price
        if isnan(self._low) or self._low > price:
            self._low = price
        self._close = price

        if prev_trade is not None and prev_trade.operation != trade.operation:
            spread = fabs(prev_trade.price - price)
            self._sum_spread += spread
            self.flips += 1
            self._avg_trade_spread = self._sum_spread / self.flips
            if isnan(self._trade_spread_min) or self._trade_spread_min > spread:
                self._trade_spread_min = spread
            if isnan(self._trade_spread_max) or self._trade_spread_max < spread:
                self._trade_spread_max = spread

        if hasattr(trade, "open_interest"):
            oi = trade.open_interest
            if self.oi_open is None:
                self.oi_open = oi
            if self.oi_high is None or self.oi_high < oi:
                self.oi_high = oi
            if self.oi_low is None or self.oi_low > oi:
                self.oi_low = oi
            self.oi_close = oi

        self.ticks += 1
        self.volume += amount
        self.money += money
        if is_buy:
            self.buy_ticks += 1
            self.buy_volume += amount
            self.buy_money += money
        else:
            self.sell_ticks += 1
            self.sell_volume += amount
            self.sell_money += money

        if prev_trade is not None and prev_trade.operation == trade.operation:
            if is_buy:
                self.conseq_buy_ticks += 1
                self.conseq_buy_volume += amount
                self.conseq_buy_money += money
            else:
                self.conseq_sell_ticks += 1
                self.conseq_sell_volume += amount
                self.conseq_sell_money += money

    cpdef finalize(self):
        cdef double sv, bv
        self._hl2 = (self._high + self._low) / 2.0
        self._hlc3 = (self._high + self._low + self._close) / 3.0
        if not isnan(self._prev_close):
            self._true_range = (
                max(self._high, self._prev_close) - min(self._low, self._prev_close)
            )
        else:
            self._true_range = self._high - self._low
        if self.ticks > 0:
            self._avg_trade_amount = self.volume / self.ticks
        if self.buy_ticks > 0:
            self._avg_buy_amount = self.buy_volume / self.buy_ticks
            self._avg_buy_money = self.buy_money / self.buy_ticks
        if self.sell_ticks > 0:
            self._avg_sell_amount = self.sell_volume / self.sell_ticks
            self._avg_sell_money = self.sell_money / self.sell_ticks
        if self.volume > 0.0:
            self._vwap = self.money / self.volume
        if self.buy_volume > 0.0:
            self._buy_vwap = self.buy_money / self.buy_volume
        if self.sell_volume > 0.0:
            self._sell_vwap = self.sell_money / self.sell_volume
        sv = self._sell_vwap
        bv = self._buy_vwap
        self.vwap_range = (sv - bv) if (not isnan(sv) and not isnan(bv)) else 0.0
        self.imbalance_ticks = self.buy_ticks - self.sell_ticks
        self.imbalance_volume = self.buy_volume - self.sell_volume
        self.imbalance_money = self.buy_money - self.sell_money
        self.imbalance_conseq_ticks = self.conseq_buy_ticks - self.conseq_sell_ticks
        self.imbalance_conseq_volume = self.conseq_buy_volume - self.conseq_sell_volume
        self.imbalance_conseq_money = self.conseq_buy_money - self.conseq_sell_money
        if self.ticks > 0:
            self._avg_trade_tick = self.imbalance_ticks / self.ticks
        return self

    def __eq__(self, other):
        if not isinstance(other, Frame):
            return NotImplemented
        return (
            self.ticks == other.ticks
            and self._open == other._open
            and self._high == other._high
            and self._low == other._low
            and self._close == other._close
            and self.volume == other.volume
        )

    def __ne__(self, other):
        result = self.__eq__(other)
        return NotImplemented if result is NotImplemented else not result

    def __repr__(self):
        return (
            f"Frame(open={self.open}, high={self.high}, low={self.low}, "
            f"close={self.close}, ticks={self.ticks}, volume={self.volume})"
        )
