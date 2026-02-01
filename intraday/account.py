from typing import Callable, Union, Any
from numbers import Real
import datetime as dt
from .report import Report, Record


class Account(object):
    def __init__(
        self,
        initial_balance: Real = 10000,
        risk_free_rate: Real = 0.0,
        convention: str = "raw",
    ):
        self.initial_balance = initial_balance
        self.position = 0
        self.position_datetime = None
        self.position_price = None
        self.position_commission = 0
        self.position_roi = 0
        self.cash = initial_balance
        self.balance = initial_balance
        self.max_balance = initial_balance
        self.min_balance = initial_balance
        self.max_drawdown = 0
        self.report = Report(
            initial_balance=initial_balance,
            risk_free_rate=risk_free_rate,
            convention=convention,
        )
        self.subscribers = {}

    def reset(self):
        self.position = 0
        self.position_datetime = None
        self.position_price = None
        self.position_commission = 0
        self.position_roi = 0
        self.cash = self.initial_balance
        self.balance = self.initial_balance
        self.max_balance = self.initial_balance
        self.min_balance = self.initial_balance
        self.max_drawdown = 0
        self.report.reset()
        self.subscribers.clear()

    def subscribe(self, who: Any, callback: Callable[..., Any]):
        self.subscribers[who] = callback

    def unsubscribe(self, who: Any):
        del self.subscribers[who]

    def on_update(self, *args, **kwargs):
        for who, callback in self.subscribers.items():
            callback(*args, **kwargs)

    def update_balance(self, price):
        self.balance = self.cash + self.position * price
        if self.position != 0 and self.position_price != 0:
            self.position_roi = (
                self.position
                * (price - self.position_price)
                / abs(self.position * self.position_price)
            )
        else:
            self.position_roi = 0
        if self.max_balance < self.balance:
            self.max_balance = self.balance
            self.min_balance = self.balance
        if self.min_balance > self.balance:
            self.min_balance = self.balance
            drawdown = self.max_balance - self.min_balance
            if self.max_drawdown < drawdown:
                self.max_drawdown = drawdown

    def close_position(
        self,
        price: Real,
        datetime: Union[Real, dt.datetime, dt.date, Any],
        commission: Real = 0,
        notes: str = None,
    ):
        if self.position == 0:
            return
        self.position_commission += commission
        record = Record(
            operation=1 if self.position > 0 else -1,
            amount=abs(self.position),
            enter_date=self.position_datetime,
            enter_price=self.position_price,
            exit_date=datetime,
            exit_price=price,
            result=self.position * (price - self.position_price)
            - self.position_commission,
            commission=self.position_commission,
            notes=notes,
        )
        self.cash = self.cash + self.position * price - commission
        self.position = 0
        self.position_datetime = None
        self.position_price = None
        self.position_commission = 0
        self.position_roi = 0
        self.update_balance(price)
        self.report.add(record)

    def update(
        self,
        datetime: Union[Real, dt.datetime, dt.date, Any],
        operation: str,
        amount: Real,
        price: Real,
        commission: Real = 0,
    ):
        assert isinstance(operation, str) and operation in "BS", ValueError(
            "Account:update: Invalid operation"
        )
        assert isinstance(amount, Real) and amount > 0, ValueError(
            "Account:update: Invalid amount"
        )
        assert isinstance(price, Real), ValueError(
            "Account:update: Invalid price"
        )
        amount = amount if operation == "B" else -amount
        new_position = self.position + amount
        if self.position == 0:
            self.position = amount
            self.position_datetime = datetime
            self.position_price = price
            self.position_commission += commission
            self.cash = self.cash - amount * price - commission
            self.update_balance(price)
        elif new_position == 0:
            self.close_position(price, datetime, commission)
        elif self.position * amount > 0:
            self.position_price = (
                self.position * self.position_price + amount * price
            ) / new_position
            self.position_commission += commission
            self.position = new_position
            self.cash = self.cash - amount * price - commission
            self.update_balance(price)
        elif self.position * new_position > 0:
            record = Record(
                operation=1 if self.position > 0 else -1,
                amount=abs(amount),
                enter_date=self.position_datetime,
                enter_price=self.position_price,
                exit_date=datetime,
                exit_price=price,
                result=(1 if self.position > 0 else -1)
                * abs(amount)
                * (price - self.position_price)
                - commission,
                commission=commission,
                notes=None,
            )
            self.cash = self.cash - amount * price - commission
            self.position = new_position
            self.update_balance(price)
            self.report.add(record)
        elif self.position * new_position < 0:
            self.close_position(price, datetime, 0)
            self.position = new_position
            self.position_datetime = datetime
            self.position_commission += commission
            self.position_price = price
            self.cash = self.cash - new_position * price - commission
            self.update_balance(price)
        return self.balance

    def __repr__(self):
        return f"{self.__class__.__name__}(initial_balance={self.initial_balance})"

    def __str__(self):
        return f"{self.__class__.__name__}{{position={self.position}, position_datetime={str(self.position_datetime)}, position_price={self.position_price}, cash={self.cash}, balance={self.balance}}}"
