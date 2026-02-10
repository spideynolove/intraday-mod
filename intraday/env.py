from numbers import Real
from typing import (
    Sequence,
    List,
    MutableSequence,
    Tuple,
    Union,
    Callable,
    Optional,
    Any,
    Literal,
)
from collections import OrderedDict
from time import sleep
from arrow import Arrow
from datetime import timedelta, datetime
import numpy as np
import gymnasium as gym
import copy
import logging
from .exchange import Exchange
from .provider import Provider, Trade
from .frame import Frame
from .processor import Processor
from .features import Feature, TradesFeature
from .actions import ActionScheme
from .rewards import RewardScheme
from .account import Account


class MultiAgentEnv(Exchange, gym.Env):
    metadata = {"render.modes": []}

    def __init__(
        self,
        provider: Union[Provider, Sequence[Provider]],
        processor: Processor,
        action_scheme: ActionScheme,
        reward_scheme: RewardScheme,
        n_agents: int = -1,
        features_pipeline: Optional[Sequence[Feature]] = None,
        initial_balance: Real = 100000,
        agent_order_delay: Union[Real, timedelta] = 3,
        broker_order_delay: Union[Real, timedelta] = 0.5,
        order_luck: float = 0.1,
        commission: Union[Real, Callable[[str, Real, Real], Real]] = 20,
        idle_penalty: Optional[float] = 0.01,
        warm_up_time: Optional[Union[Real, timedelta]] = 10 * 60,
        delay_per_second: Optional[float] = None,
        instant_balance_update=False,
        max_frames_period=100,
        max_trades_period=1000,
        log: Optional[logging.Logger] = None,
        **kwargs,
    ):
        assert isinstance(n_agents, int) and (n_agents > 0 or n_agents == -1)
        self.n_agents = n_agents
        assert isinstance(initial_balance, Real) and initial_balance > 0
        self.initial_balance = initial_balance
        assert idle_penalty is None or isinstance(idle_penalty, float)
        self.idle_penalty = idle_penalty
        if warm_up_time is None:
            warm_up_time = timedelta(seconds=0.0)
        elif isinstance(warm_up_time, Real):
            warm_up_time = timedelta(seconds=float(warm_up_time))
        elif isinstance(warm_up_time, timedelta):
            pass
        else:
            raise ValueError("Invalid warm_up_time value")
        assert warm_up_time.total_seconds() >= 0
        self.warm_up_time: timedelta = warm_up_time
        assert (
            delay_per_second is None
            or isinstance(delay_per_second, float)
            and 0.0 <= delay_per_second < 1.0
        )
        self.delay_per_second = delay_per_second
        accounts = [
            Account(initial_balance=initial_balance) for _ in range(n_agents)
        ]
        self._log = (
            log
            if isinstance(log, logging.Logger)
            else logging.getLogger(self.__class__.__name__)
        )
        super().__init__(
            accounts=accounts,
            agent_order_delay=agent_order_delay,
            broker_order_delay=broker_order_delay,
            order_luck=order_luck,
            commission=commission,
            instant_balance_update=instant_balance_update,
            **kwargs,
        )
        if isinstance(provider, Provider):
            self.providers = [provider]
        elif isinstance(provider, Sequence):
            assert (
                len(provider) > 0
            ), "You should specify at least 1 data provider!"
            assert all(
                [isinstance(p, Provider) for p in provider]
            ), "Some of objects are not providers!"
            self.providers = provider
        else:
            raise ValueError("Invalid provider argument")
        self.provider: Optional[Provider] = None
        self.processor = processor
        self.action_scheme = action_scheme
        self.reward_scheme = reward_scheme
        self.features_pipeline = features_pipeline
        self.trades_features = []
        spaces = OrderedDict()
        if isinstance(features_pipeline, Sequence):
            for feature in features_pipeline:
                if isinstance(feature, TradesFeature):
                    self.trades_features.append(feature)
                    trades_period = feature.trades_period
                    if isinstance(trades_period, int):
                        pass
                    elif isinstance(trades_period, Sequence):
                        trades_period = max(trades_period)
                    else:
                        trades_period = 0
                    if (
                        max_trades_period is None
                        or max_trades_period < trades_period
                    ):
                        max_trades_period = trades_period
                for name, space in feature.spaces.items():
                    spaces[name] = space
                frames_period = feature.period
                if isinstance(frames_period, int):
                    pass
                elif isinstance(frames_period, Sequence):
                    frames_period = max(frames_period)
                else:
                    frames_period = 0
                if (
                    max_frames_period is None
                    or max_frames_period < frames_period
                ):
                    max_frames_period = frames_period
        spaces["position"] = gym.spaces.Box(-np.inf, np.inf, (1,))
        spaces["position_roi"] = gym.spaces.Box(-np.inf, np.inf, (1,))
        spaces["profit_factor"] = gym.spaces.Box(0, np.inf, (1,))
        spaces["sortino_ratio"] = gym.spaces.Box(-np.inf, np.inf, (1,))
        self.max_trades_period = max(1000, max_trades_period or 0)
        self.max_frames_period = max(100, max_frames_period or 0)
        self.observation_space: gym.spaces.Space = gym.spaces.Dict(spaces)
        self.observation_names = tuple(spaces.keys())
        self.action_space: gym.spaces.Space = action_scheme.space
        self.episode_start_datetime: Optional[Union[Arrow, datetime]] = None
        self.episode_max_duration: Optional[timedelta] = None
        self.span_start_time: Optional[Union[Arrow, datetime]] = None
        self.trades: List[Trade] = []
        self.frames: List[Frame] = []
        self.state: Optional[OrderedDict] = None
        self.rng: Optional[np.random.RandomState] = None
        self.seed()

    def seed(self, seed: Optional[Any] = None) -> Any:
        from gymnasium.utils import seeding

        self.rng, seed = seeding.np_random(seed)
        return seed

    def reset(
        self,
        episode_start_datetime: Optional[Union[Arrow, datetime]] = None,
        episode_min_duration: Optional[Union[Real, timedelta]] = None,
        episode_max_duration: Optional[Union[Real, timedelta]] = timedelta(
            hours=2
        ),
        n_agents: int = -1,
        keep_state=False,
        **kwargs,
    ) -> Sequence[OrderedDict]:
        if episode_start_datetime is None:
            pass
        elif isinstance(episode_start_datetime, (Arrow, datetime)):
            pass
        else:
            raise ValueError("Invalid episode_start_datetime value")
        if episode_min_duration is None:
            pass
        elif isinstance(episode_min_duration, Real):
            pass
        elif isinstance(episode_min_duration, timedelta):
            pass
        else:
            raise ValueError("Invalid episode_min_duration value")
        if episode_max_duration is None:
            pass
        elif isinstance(episode_max_duration, Real):
            episode_max_duration = timedelta(
                seconds=float(episode_max_duration)
            )
        elif isinstance(episode_max_duration, timedelta):
            pass
        else:
            raise ValueError("Invalid episode_max_duration value")
        assert (
            episode_max_duration is None
            or episode_max_duration.total_seconds() > 0
        )
        if not keep_state:
            super().reset()
            self.processor.reset()
            self.action_scheme.reset()
            self.reward_scheme.reset()
            for feature in self.features_pipeline:
                feature.reset()
            self.episode_start_datetime = None
            self.episode_max_duration = None
            self.span_start_time = None
            self.trades.clear()
            self.frames.clear()
        if self.n_agents <= 0 and n_agents <= 0:
            raise ValueError(
                "You should specify n_agents either at env creation or in a reset() call"
            )
        n_agents = max(self.n_agents, n_agents)
        if len(self.accounts) < n_agents:
            new_accounts = [
                Account(initial_balance=self.initial_balance)
                for _ in range(n_agents - len(self.accounts))
            ]
            self.accounts.extend(new_accounts)
        if len(self.providers) == 1:
            self.provider = self.providers[0]
        else:
            if self.provider is not None:
                self.provider.close()
            self.provider = self.rng.choice(self.providers)
        self.episode_start_datetime = (
            self.provider.reset(
                episode_start_datetime=(
                    episode_start_datetime - self.warm_up_time
                    if episode_start_datetime is not None
                    else None
                ),
                episode_min_duration=(
                    episode_min_duration + self.warm_up_time
                    if episode_min_duration is not None
                    else None
                ),
                rng=self.rng,
                **kwargs,
            )
            + self.warm_up_time
        )
        self.episode_max_duration = episode_max_duration
        if not keep_state:
            action = self.action_scheme.get_random_action()
            for account in self.accounts:
                self.action_scheme.process_action(
                    exchange=self,
                    account=account,
                    action=action,
                    time=self.episode_start_datetime,
                )
        self.span_start_time = Arrow.now()
        frame, done = self._get_next_frame()
        if frame is None or done:
            raise RuntimeError(
                f"Failed to get initial state for {self.episode_start_datetime}"
            )
        states = self._make_states()
        return states

    def step(self, actions: Sequence[Any], **kwargs) -> Tuple[
        Union[Sequence[OrderedDict], None],
        Sequence[float],
        bool,
        Union[Frame, None],
    ]:
        assert isinstance(actions, Sequence) and len(actions) == len(
            self.accounts
        )
        action_time = self.last_trade.datetime + self.agent_order_delay
        for account, action in zip(self.accounts, actions):
            self.action_scheme.process_action(
                exchange=self, account=account, action=action, time=action_time
            )
        frame, done = self._get_next_frame()
        if frame is not None:
            states = self._make_states()
            done = done and states is not None
        else:
            done = True
            states = None
        if self.idle_penalty is not None:
            penalty = self.idle_penalty * abs(frame.close - frame.open)
            for account in self.accounts:
                if account.position == 0:
                    account.balance -= penalty
        if (
            not done
            and self.episode_max_duration is not None
            and frame.time_end is not None
        ):
            episode_duration = frame.time_end - self.episode_start_datetime
            done = episode_duration >= self.episode_max_duration
        if done:
            for account in self.accounts:
                if account.position != 0:
                    operation = "S" if account.position > 0 else "B"
                    amount = abs(account.position)
                    price = self.last_trade.price
                    commission = self._get_commission(operation, amount, price)
                    account.close_position(
                        price, self.last_trade.datetime, commission
                    )
        elif not self.instant_balance_update:
            self._update_balances(
                price=self.last_trade.price, dt=self.last_trade.datetime
            )
        rewards = []
        for account in self.accounts:
            rewards.append(
                self.reward_scheme.get_reward(env=self, account=account)
            )
        return states, rewards, done, frame

    def _get_next_frame(self) -> Tuple[Union[Frame, None], bool]:
        frame = None
        done = False
        while True:
            try:
                trade = next(self.provider)
                self.process_trade(trade)
                if (
                    trade.datetime
                    >= self.episode_start_datetime - self.warm_up_time
                ):
                    self.trades.append(trade)
                    if len(self.trades) > self.max_trades_period + 1:
                        del self.trades[0]
                    frame = self.processor.process(self.trades)
                    for feature in self.trades_features:
                        feature.update(self.trades)
                if self.delay_per_second is not None:
                    span_duration = float(
                        (Arrow.now() - self.span_start_time).microseconds
                    )
                    if span_duration >= 1000000.0 * (
                        1.0 - self.delay_per_second
                    ):
                        sleep(self.delay_per_second)
                        self.span_start_time = Arrow.now()
                if frame is not None:
                    self._process_frame(frame)
                    if (
                        frame.time_end is not None
                        and frame.time_end >= self.episode_start_datetime
                    ):
                        break
            except StopIteration:
                frame = self.processor.finish()
                if frame is not None:
                    self._process_frame(frame)
                done = True
                break
        return frame, done

    def _process_frame(self, frame: Frame):
        self.frames.append(frame)
        if len(self.frames) > self.max_frames_period + 1:
            del self.frames[0]
        self.state = OrderedDict()
        for feature in self.features_pipeline:
            feature.process(self.frames, self.state)

    def _make_states(self) -> MutableSequence[OrderedDict]:
        assert isinstance(self.state, OrderedDict)
        agent_states = []
        for account in self.accounts:
            agent_state = (
                copy.copy(self.state) if self.n_agents > 1 else self.state
            )
            agent_state["position"] = account.position
            agent_state["position_roi"] = account.position_roi
            agent_state["profit_factor"] = account.report.profit_factor or 0.0
            agent_state["sortino_ratio"] = account.report.sortino_ratio or 0.0
            agent_states.append(agent_state)
        return agent_states

    def close(self):
        super().reset()
        self.provider.close()
        self.provider = None
        self.processor.reset()
        self.action_scheme.reset()
        self.reward_scheme.reset()
        self.episode_start_datetime = None
        self.episode_max_duration = None
        self.span_start_time = None
        self.trades.clear()
        self.frames.clear()

    def render(
        self, mode: Literal["human", "rgb_array", "ansi"] = "human"
    ) -> Union[None, np.ndarray, str]:
        raise NotImplementedError()

    def __repr__(self):
        return f"MultiAgentEnv(provider={self.provider.__class__.__name__}, processor={self.processor.__class__.__name__})"

    def __str__(self):
        return self.__repr__()


class SingleAgentEnv(MultiAgentEnv):
    metadata = {"render.modes": ["human", "rgb_array"]}
    Window_Width = 600
    Window_Height = 400
    Margin_Top = 0
    Margin_Bottom = 5
    Margin_Side = 5
    Margin_Inter = 12
    Bar_Width = 5
    Bar_Space = 1
    Price_Margin = 0.05
    Clr_Background = 0.9, 0.9, 0.9
    Clr_Bar_Rise = 0.1, 0.9, 0.1
    Clr_Bar_Fall = 1.0, 0.2, 0.2
    Clr_Pos_Long = 0.1, 0.9, 0.1
    Clr_Pos_Short = 1.0, 0.2, 0.2
    Clr_Pos_None = 0.2, 0.2, 0.2
    Clr_Text = 50, 50, 50, 255

    def __init__(
        self,
        provider: Union[Provider, Sequence[Provider]],
        processor: Processor,
        action_scheme: ActionScheme,
        reward_scheme: RewardScheme,
        initial_balance: Real = 100000,
        agent_order_delay: Union[Real, timedelta] = 3,
        broker_order_delay: Union[Real, timedelta] = 0.5,
        order_luck=0.1,
        commission: Union[Real, Callable[[int, Real, Real], Real]] = 20,
        idle_penalty: Optional[float] = 0.01,
        warm_up_time: Optional[Union[Real, timedelta]] = 10 * 60,
        delay_per_second: Optional[float] = None,
        instant_balance_update=False,
        log: Optional[logging.Logger] = None,
        **kwargs,
    ):
        super().__init__(
            n_agents=1,
            provider=provider,
            processor=processor,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            initial_balance=initial_balance,
            agent_order_delay=agent_order_delay,
            broker_order_delay=broker_order_delay,
            order_luck=order_luck,
            commission=commission,
            idle_penalty=idle_penalty,
            warm_up_time=warm_up_time,
            delay_per_second=delay_per_second,
            instant_balance_update=instant_balance_update,
            log=log,
            **kwargs,
        )
        self.account = self.accounts[0]
        self.viewer: Optional[object] = None
        self.area_price: Optional[Tuple[float, float, float, float]] = None
        self.area_score: Optional[Tuple[float, float, float, float]] = None
        self.caption_label: Optional[object] = None
        self.episode_label: Optional[object] = None
        self.status_label: Optional[object] = None
        self.price_chart: Optional[object] = None
        self.score_chart: Optional[object] = None
        self.score_chart2: Optional[object] = None
        self.trans_price: Optional[object] = None
        self.scale_price: Optional[object] = None
        self.trans_score: Optional[object] = None
        self.scale_score: Optional[object] = None
        self.min_x: Optional[float] = None
        self.max_x: Optional[float] = None
        self.min_p: Optional[float] = None
        self.max_p: Optional[float] = None
        self.min_s: Optional[float] = None
        self.max_s: Optional[float] = None
        self.last_position = 0
        self.score = 0.0
        self.n_bars = 0

    def reset(
        self,
        episode_start_datetime: Optional[Union[Arrow, datetime]] = None,
        episode_min_duration: Optional[Union[Real, timedelta]] = None,
        episode_max_duration: Optional[Union[Real, timedelta]] = timedelta(
            hours=2
        ),
        keep_state=False,
        **kwargs,
    ) -> OrderedDict:
        states = super().reset(
            episode_start_datetime=episode_start_datetime,
            episode_min_duration=episode_min_duration,
            episode_max_duration=episode_max_duration,
            keep_state=keep_state,
            **kwargs,
        )
        if self.viewer is not None:
            (
                self.min_x,
                self.max_x,
                self.min_p,
                self.max_p,
                self.min_s,
                self.max_s,
            ) = (None,) * 6
            self.price_chart.reset()
            self.score_chart.reset()
            self.last_position = self.account.position
            self.score = 0.0
            self.n_bars = 0
            self.caption_label.text = (
                self.provider.name + " " + self.processor.name
            )
            self.episode_label.text = (
                self.provider.episode_start_datetime.strftime(
                    "%Y.%m.%d %H:%M:%S%z"
                )
            )
        return states[0] if states is not None else None

    def step(
        self, action: Any, **kwargs
    ) -> Tuple[Union[OrderedDict, None], float, bool, Union[Frame, None]]:
        states, rewards, done, frame = super().step([action], **kwargs)
        state = states[0] if states is not None else None
        reward = rewards[0] if rewards is not None else None
        done = done or self.account in self.halted_accounts
        if self.viewer is not None:
            self._update_chart(self.frames[-1], reward)
        return state, reward, done, frame

    def render(self, mode="rgb_array"):
        from .render import (
            Viewer,
            Group,
            Label,
            Translate,
            Scale,
            FilledPolygon,
            LineWidth,
        )

        if self.viewer is None:
            self.viewer = Viewer(
                self.Window_Width, self.Window_Height, resizable=True
            )
            width, height = self.viewer.width, self.viewer.height
            self._update_layout(width, height)
            for a in (self.area_price, self.area_score):
                box = FilledPolygon(
                    [(a[0], a[1]), (a[0], a[3]), (a[2], a[3]), (a[2], a[1])],
                    color=self.Clr_Background,
                )
                self.viewer.add_geom(box)
            self.trans_price = Translate(0, 0)
            self.scale_price = Scale(1, 1)
            self.price_chart = Group()
            self.price_chart.add_attr(self.scale_price)
            self.price_chart.add_attr(self.trans_price)
            self.price_chart.add_attr(LineWidth(2))
            self.viewer.add_geom(self.price_chart)
            self.trans_score = Translate(0, 0)
            self.scale_score = Scale(1, 1)
            self.score_chart = Group()
            self.score_chart.add_attr(self.scale_score)
            self.score_chart.add_attr(self.trans_score)
            self.score_chart.add_attr(LineWidth(2))
            self.viewer.add_geom(self.score_chart)
            self.caption_label = Label(
                self.provider.name + " " + self.processor.name,
                (width // 2, self.area_price[3]),
                anchor_x="center",
                anchor_y="top",
                align="center",
                font_name="Arial",
                font_size=10,
                bold=True,
                color=self.Clr_Text,
            )
            self.viewer.add_geom(self.caption_label)
            self.episode_label = Label(
                self.provider.episode_start_datetime.strftime(
                    "%Y.%m.%d %H:%M:%S%z"
                ),
                (
                    self.Margin_Side,
                    self.area_score[3] + self.Margin_Inter / 2 + 1,
                ),
                anchor_x="left",
                anchor_y="center",
                align="left",
                font_name="Arial",
                font_size=10,
                bold=True,
                color=self.Clr_Text,
            )
            self.viewer.add_geom(self.episode_label)
            self.status_label = Label(
                "status",
                (
                    width - self.Margin_Side,
                    self.area_score[3] + self.Margin_Inter / 2 + 1,
                ),
                anchor_x="right",
                anchor_y="center",
                align="right",
                font_name="Arial",
                font_size=10,
                bold=True,
                color=self.Clr_Text,
            )
            self.viewer.add_geom(self.status_label)
            self._update_chart(self.frames[-1], 0)
        elif self.n_bars > 0:
            self._update_layout(self.viewer.width, self.viewer.height)
            scale_x = (self.area_price[2] - self.area_price[0]) / (
                self.max_x - self.min_x + 2 * self.Bar_Width
            )
            scale_x = min(1.0, scale_x)
            if self.max_p > self.min_p:
                scale_p = (self.area_price[3] - self.area_price[1]) / (
                    (self.max_p - self.min_p) * (1 + 2 * self.Price_Margin)
                )
            else:
                scale_p = self.area_price[1] / self.min_p
            self.trans_price.set_translation(
                (self.Bar_Width - self.min_x) * scale_x + self.area_price[0],
                ((self.max_p - self.min_p) * self.Price_Margin - self.min_p)
                * scale_p
                + self.area_price[1],
            )
            self.scale_price.set_scale(scale_x, scale_p)
            if self.max_s > self.min_s:
                scale_s = (self.area_score[3] - self.area_score[1]) / (
                    (self.max_s - self.min_s) * (1 + 2 * self.Price_Margin)
                )
            else:
                scale_s = 1.0
            self.trans_score.set_translation(
                (self.Bar_Width - self.min_x) * scale_x + self.area_score[0],
                ((self.max_s - self.min_s) * self.Price_Margin - self.min_s)
                * scale_s
                + self.area_score[1],
            )
            self.scale_score.set_scale(scale_x, scale_s)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def _update_layout(self, width, height):
        self.area_price = (
            self.Margin_Side,
            height * 1 / 3 + self.Margin_Inter / 2,
            width - self.Margin_Side,
            height - self.Margin_Top,
        )
        self.area_score = (
            self.Margin_Side,
            self.Margin_Bottom,
            width - self.Margin_Side,
            height * 1 / 3 - self.Margin_Inter / 2,
        )

    def _update_chart(self, frame: Frame, reward: Real):
        from .render import Line

        n = self.n_bars
        xc = float((self.Bar_Width + self.Bar_Space) * n + self.Bar_Width / 2)
        xl, xr = xc - self.Bar_Width / 2, xc + self.Bar_Width / 2
        o, h, l, c = frame.open, frame.high, frame.low, frame.close
        bar_clr = self.Clr_Bar_Rise if c >= o else self.Clr_Bar_Fall
        vert = Line((xc, l), (xc, h), color=bar_clr)
        left = Line((xl, o), (xc, o), color=bar_clr)
        right = Line((xr, c), (xc, c), color=bar_clr)
        self.price_chart.add_geom(left)
        self.price_chart.add_geom(right)
        self.price_chart.add_geom(vert)
        bal_clr = (
            self.Clr_Pos_Long
            if self.account.position > 0
            else (
                self.Clr_Pos_Short
                if self.account.position < 0
                else self.Clr_Pos_None
            )
        )
        bal_seg = Line(
            (xc - self.Bar_Width - self.Bar_Space, self.score),
            (xc, self.score + reward),
            color=bal_clr,
        )
        self.score_chart.add_geom(bal_seg)
        self.last_position = self.account.position
        self.score += reward
        self.status_label.text = f"{self.score:0.2f}    {self.account.balance or np.nan:0.2f}   PF: {self.account.report.profit_factor or np.nan:0.2f}  SR: {self.account.report.sortino_ratio or np.nan:0.2f}  "
        if self.score > 0.0:
            self.status_label.color = self.Clr_Bar_Rise
        else:
            self.status_label.color = self.Clr_Bar_Fall
        if self.min_x is None or self.min_x > xl:
            self.min_x = xl
        if self.max_x is None or self.max_x < xr:
            self.max_x = xr
        if self.min_p is None or self.min_p > l:
            self.min_p = l
        if self.max_p is None or self.max_p < h:
            self.max_p = h
        if self.min_s is None or self.min_s > self.score:
            self.min_s = self.score
        if self.max_s is None or self.max_s < self.score:
            self.max_s = self.score
        self.n_bars += 1

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
