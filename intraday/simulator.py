import warnings
from typing import List, Tuple
import numpy as np
from .provider import Trade, Candle, Kline, Provider


class Simulator(Provider):
    def __init__(self, spread: float = 0.0005, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(spread, float) and 0 <= spread <= 1.0
        self.spread = spread

    def simulate_trades_from_candle(self, candle: Candle) -> List[Trade]:
        open, high, low, close, volume = (
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
        )
        if volume <= 0:
            return []
        step = self.spread * (high + low) / 2
        if np.random.random() < 0.5:
            prices, orders = self._zigzag_open_high_low_close(
                open, high, low, close, step
            )
        else:
            prices, orders = self._zigzag_open_low_high_close(
                open, high, low, close, step
            )
        amounts = np.ones_like(prices) * volume / len(prices)
        result = []
        t = candle.time_start
        dt = (candle.time_end - candle.time_start) / len(prices)
        for order, price, amount in zip(orders, prices, amounts):
            result.append(
                Trade(
                    datetime=t,
                    operation="B" if order > 0 else "S",
                    amount=amount,
                    price=price,
                )
            )
            t += dt
        return result

    def simulate_trades_from_kline(self, kline: Kline) -> List[Trade]:
        open, high, low, close = kline.open, kline.high, kline.low, kline.close
        volume, money, buy_volume, buy_money = (
            kline.volume,
            kline.money,
            kline.buy_volume,
            kline.buy_money,
        )
        sell_volume, sell_money = volume - buy_volume, money - buy_money
        if volume <= 0:
            return []
        step = self.spread * (high + low) / 2
        if np.random.random() < 0.5:
            prices, orders = self._zigzag_open_high_low_close(
                open, high, low, close, step
            )
        else:
            prices, orders = self._zigzag_open_low_high_close(
                open, high, low, close, step
            )
        amounts = np.zeros_like(prices)
        if buy_volume <= 0:
            orders[:] = 1
        elif sell_volume <= 0:
            orders[:] = -1
        if buy_volume > 0:
            buy_vwap = buy_money / buy_volume
            amounts[orders > 0] = buy_volume * self._target_coefficients(
                prices[orders > 0], target=buy_vwap
            )
        if sell_volume > 0:
            sell_vwap = sell_money / sell_volume
            amounts[orders < 0] = sell_volume * self._target_coefficients(
                prices[orders < 0], target=sell_vwap
            )
        result = []
        t = kline.time_start
        dt = (kline.time_end - kline.time_start) / len(prices)
        for order, price, amount in zip(orders, prices, amounts):
            result.append(
                Trade(
                    datetime=t,
                    operation="B" if order > 0 else "S",
                    amount=amount,
                    price=price,
                )
            )
            t += dt
        return result

    @staticmethod
    def _zigzag_open_high_low_close(
        open, high, low, close, step
    ) -> Tuple[np.ndarray, np.ndarray]:
        open_to_high = np.concatenate((np.arange(open, high, step), (high,)))
        high_to_low = np.concatenate(
            (np.arange(high - step, low, -step), (low,))
        )
        low_to_close = np.concatenate(
            (np.arange(low + step, close, step), (close,))
        )
        prices = np.concatenate((open_to_high, high_to_low, low_to_close))
        orders = np.concatenate(
            (
                np.ones(len(open_to_high), dtype=np.int8),
                -1 * np.ones(len(high_to_low), dtype=np.int8),
                np.ones(len(low_to_close), dtype=np.int8),
            )
        )
        return prices, orders

    @staticmethod
    def _zigzag_open_low_high_close(
        open, high, low, close, step
    ) -> Tuple[np.ndarray, np.ndarray]:
        open_to_low = np.concatenate((np.arange(open, low, -step), (low,)))
        low_to_high = np.concatenate(
            (np.arange(low + step, high, step), (high,))
        )
        high_to_close = np.concatenate(
            (np.arange(high - step, close, -step), (close,))
        )
        prices = np.concatenate((open_to_low, low_to_high, high_to_close))
        orders = np.concatenate(
            (
                -1 * np.ones(len(open_to_low), dtype=np.int8),
                np.ones(len(low_to_high), dtype=np.int8),
                -1 * np.ones(len(high_to_close), dtype=np.int8),
            )
        )
        return prices, orders

    @staticmethod
    def _target_coefficients(prices: np.ndarray, target: float) -> np.ndarray:
        result = np.ones_like(prices, dtype=np.float32)
        indices = np.argsort(prices)
        sorted_prices = prices[indices]
        target_index = max(1, np.searchsorted(sorted_prices, target))
        prices_below = sorted_prices[:target_index]
        prices_above = sorted_prices[target_index:]
        n_below, n_above = len(prices_below), len(prices_above)
        if n_below == 0 or n_above == 0:
            warnings.warn(
                "Target price is outside of the given prices range!",
                source=Simulator._target_coefficients,
            )
        else:
            mean_below, mean_above = prices_below.mean(), prices_above.mean()
            k_below = (mean_above - target) / (mean_above - mean_below)
            k_above = (target - mean_below) / (mean_above - mean_below)
            result[indices[:target_index]] = k_below / n_below
            result[indices[target_index:]] = k_above / n_above
        result = result / result.sum()
        return result
