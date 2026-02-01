from datetime import datetime
from typing import Sequence
from abc import ABC, abstractmethod
from collections import defaultdict
from numbers import Real
import numpy as np
from gymnasium import spaces
from .account import Account
from .exchange import Exchange, MarketOrder, StopOrder, TakeProfitOrder


class ActionScheme(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def get_random_action(self):
        raise NotImplementedError()

    @abstractmethod
    def get_default_action(self):
        raise NotImplementedError()

    @abstractmethod
    def process_action(
        self, exchange: Exchange, account: Account, action, time: datetime
    ):
        raise NotImplementedError()

    @property
    @abstractmethod
    def space(self) -> spaces.Space:
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class BuySellCloseAction(ActionScheme):
    space = spaces.Discrete(3)

    def __init__(self, amount: Real = 1, **kwargs):
        super().__init__(**kwargs)
        self.amount = abs(amount)

    def reset(self):
        pass

    def get_random_action(self) -> int:
        return np.random.randint(0, 3)

    def get_default_action(self) -> int:
        return 2

    def process_action(
        self, exchange: Exchange, account: Account, action, time: datetime
    ):
        assert 0 <= action <= 2
        target_position = (
            self.amount if action == 0 else -self.amount if action == 1 else 0
        )
        delta = target_position - account.position
        if delta != 0:
            order = MarketOrder(
                account=account,
                operation="B" if delta > 0 else "S",
                amount=abs(delta),
                time_init=time,
                time_kill=None,
            )
            exchange.add_order(order)


class PingPongAction(ActionScheme):
    space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.position = defaultdict(lambda: None)
        self.lower_price = defaultdict(lambda: None)
        self.upper_price = defaultdict(lambda: None)
        self.trail_delta = defaultdict(lambda: None)
        self.stop_delta = defaultdict(lambda: None)
        self.buy_order_id = defaultdict(lambda: None)
        self.sell_order_id = defaultdict(lambda: None)
        self.buy_stop_order_id = defaultdict(lambda: None)
        self.sell_stop_order_id = defaultdict(lambda: None)

    def reset(self):
        self.position.clear()
        self.lower_price.clear()
        self.upper_price.clear()
        self.trail_delta.clear()
        self.stop_delta.clear()
        self.buy_order_id.clear()
        self.sell_order_id.clear()
        self.buy_stop_order_id.clear()
        self.sell_stop_order_id.clear()

    def get_random_action(self) -> np.ndarray:
        return np.array((-np.inf, np.inf, 0, 0))

    def get_default_action(self) -> np.ndarray:
        return np.array((-np.inf, np.inf, 0, 0))

    def process_action(
        self, exchange: Exchange, account: Account, action, time: datetime
    ):
        assert isinstance(action, (Sequence, np.ndarray)) and len(action) == 4
        lower_price, upper_price, trail_delta, stop_delta = action
        if self.lower_price[account] is None:
            account.subscribe(self, lambda ex, acc, t: self.update(ex, acc, t))
        position_changed = (
            self.position[account] is None
            or self.position[account] != account.position
        )
        lower_price_changed = (
            self.lower_price[account] is None
            or self.lower_price[account] != lower_price
        )
        upper_price_changed = (
            self.upper_price[account] is None
            or self.upper_price[account] != upper_price
        )
        trail_delta_changed = (
            self.trail_delta[account] is None
            or self.trail_delta[account] != trail_delta
        )
        stop_delta_changed = (
            self.stop_delta[account] is None
            or self.stop_delta[account] != stop_delta
        )
        self.lower_price[account] = lower_price
        self.upper_price[account] = upper_price
        self.trail_delta[account] = trail_delta
        self.stop_delta[account] = stop_delta
        self.position[account] = account.position
        if account.position == 0:
            if lower_price_changed or trail_delta_changed or position_changed:
                self._open_buy(exchange, account, time)
            if upper_price_changed or trail_delta_changed or position_changed:
                self._open_sell(exchange, account, time)
        elif account.position > 0:
            if upper_price_changed or trail_delta_changed or position_changed:
                self._open_sell(exchange, account, time)
            if upper_price_changed or stop_delta_changed or position_changed:
                self._open_stop_sell(exchange, account, time)
        elif account.position < 0:
            if lower_price_changed or trail_delta_changed or position_changed:
                self._open_buy(exchange, account, time)
            if lower_price_changed or stop_delta_changed or position_changed:
                self._open_stop_buy(exchange, account, time)

    def update(self, exchange: Exchange, account: Account, time: datetime):
        old_position = self.position[account] or 0
        new_position = account.position
        if old_position < 0 and new_position == 0:
            self._kill_stop_buy(exchange, account, time)
        elif old_position < 0 and new_position > 0:
            self._kill_stop_buy(exchange, account, time)
            self._open_stop_sell(exchange, account, time)
            self._open_sell(exchange, account, time)
        elif old_position > 0 and new_position == 0:
            self._kill_stop_sell(exchange, account, time)
        elif old_position > 0 and new_position < 0:
            self._kill_stop_sell(exchange, account, time)
            self._open_stop_buy(exchange, account, time)
            self._open_buy(exchange, account, time)
        elif old_position == 0 and new_position < 0:
            self._open_stop_buy(exchange, account, time)
            self._open_buy(exchange, account, time)
        elif old_position == 0 and new_position > 0:
            self._open_stop_sell(exchange, account, time)
            self._open_sell(exchange, account, time)
        elif old_position == 0 and new_position == 0:
            pass
        self.position[account] = new_position

    def _open_sell(self, exchange: Exchange, account: Account, time: datetime):
        if 1 + account.position > 0:
            self.sell_order_id[account] = exchange.replace_order(
                id=self.sell_order_id[account],
                new_order=TakeProfitOrder(
                    account=account,
                    operation="S",
                    amount=1 + account.position,
                    time_init=time,
                    time_kill=None,
                    target_price=self.upper_price[account],
                    trail_delta=self.trail_delta[account],
                    best_price=None,
                ),
            )

    def _open_buy(self, exchange: Exchange, account: Account, time: datetime):
        if 1 - account.position > 0:
            self.buy_order_id[account] = exchange.replace_order(
                id=self.buy_order_id[account],
                new_order=TakeProfitOrder(
                    account=account,
                    operation="B",
                    amount=1 - account.position,
                    time_init=time,
                    time_kill=None,
                    target_price=self.lower_price[account],
                    trail_delta=self.trail_delta[account],
                    best_price=None,
                ),
            )

    def _kill_buy(self, exchange: Exchange, account: Account, time: datetime):
        if self.buy_order_id[account] is not None:
            exchange.kill_order(self.buy_order_id[account], time_kill=time)
            self.buy_order_id[account] = None

    def _kill_sell(self, exchange: Exchange, account: Account, time: datetime):
        if self.sell_order_id[account] is not None:
            exchange.kill_order(self.sell_order_id[account], time_kill=time)
            self.sell_order_id[account] = None

    def _open_stop_buy(
        self, exchange: Exchange, account: Account, time: datetime
    ):
        self.buy_stop_order_id[account] = exchange.replace_order(
            id=self.buy_stop_order_id[account],
            new_order=StopOrder(
                account=account,
                operation="B",
                amount=abs(account.position),
                price=account.position_price + self.stop_delta[account],
                time_init=time,
                time_kill=None,
            ),
        )

    def _open_stop_sell(
        self, exchange: Exchange, account: Account, time: datetime
    ):
        self.sell_stop_order_id[account] = exchange.replace_order(
            id=self.sell_stop_order_id[account],
            new_order=StopOrder(
                account=account,
                operation="S",
                amount=account.position,
                price=account.position_price - self.stop_delta[account],
                time_init=time,
                time_kill=None,
            ),
        )

    def _kill_stop_buy(
        self, exchange: Exchange, account: Account, time: datetime
    ):
        if self.buy_stop_order_id[account] is not None:
            exchange.kill_order(
                self.buy_stop_order_id[account], time_kill=time
            )
            self.buy_stop_order_id[account] = None

    def _kill_stop_sell(
        self, exchange: Exchange, account: Account, time: datetime
    ):
        if self.sell_stop_order_id[account] is not None:
            exchange.kill_order(
                self.sell_stop_order_id[account], time_kill=time
            )
            self.sell_stop_order_id[account] = None
