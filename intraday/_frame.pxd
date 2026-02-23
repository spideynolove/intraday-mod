cdef class Frame:
    cdef public object time_start
    cdef public object time_end
    cdef public double duration
    cdef public double _open, _high, _low, _close, _prev_close
    cdef public double _hl2, _hlc3, _true_range
    cdef public int flips
    cdef public double _sum_spread
    cdef public double _avg_trade_spread
    cdef public double _trade_spread_min, _trade_spread_max
    cdef public int ticks, buy_ticks, sell_ticks
    cdef public double volume, money, buy_volume, buy_money, sell_volume, sell_money
    cdef public double _vwap, _buy_vwap, _sell_vwap
    cdef public double _avg_trade_tick, _avg_trade_amount
    cdef public double _avg_buy_amount, _avg_buy_money
    cdef public double _avg_sell_amount, _avg_sell_money
    cdef public int conseq_buy_ticks, conseq_sell_ticks
    cdef public double conseq_buy_volume, conseq_buy_money
    cdef public double conseq_sell_volume, conseq_sell_money
    cdef public double vwap_range
    cdef public int imbalance_ticks, imbalance_conseq_ticks
    cdef public double imbalance_volume, imbalance_money
    cdef public double imbalance_conseq_volume, imbalance_conseq_money
    cdef public object oi_open, oi_high, oi_low, oi_close
    cdef dict __dict__

    cpdef update(self, trades)
    cpdef finalize(self)
