from typing import Sequence, Literal
from collections import OrderedDict
import math
import gymnasium as gym
import pandas as pd
import arrow
from intraday.frame import Frame
from intraday.feature import Feature


IMPACT_WEIGHTS = {1: 0.2, 2: 0.6, 3: 1.0}

EVENT_CATEGORIES = {
    'employment': 1,
    'inflation': 2,
    'rate': 3,
    'gdp': 4,
    'trade': 5,
    'other': 0,
}

CATEGORY_KEYWORDS = {
    'employment': ['nfp', 'non-farm', 'unemployment', 'jobless', 'employment', 'payroll'],
    'inflation': ['cpi', 'ppi', 'inflation', 'pcr', 'pce'],
    'rate': ['rate', 'fomc', 'fed', 'boe', 'ecb', 'rba', 'boj', 'snb'],
    'gdp': ['gdp', 'growth'],
    'trade': ['trade balance', 'current account'],
}


def _classify_event(name: str) -> int:
    lower = name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(k in lower for k in keywords):
            return EVENT_CATEGORIES[category]
    return EVENT_CATEGORIES['other']


class CalendarEvents(Feature):
    def __init__(
        self,
        calendar_path: str,
        currencies: list[str],
        pre_event_window: int = 60,
        post_event_window: int = 30,
        decay_halflife: int = 30,
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to)
        self.pre_event_window = pre_event_window
        self.post_event_window = post_event_window
        self.decay_halflife = decay_halflife
        self._decay_lambda = math.log(2) / decay_halflife
        self.currencies = [c.upper() for c in currencies]

        self.names = [
            'minutes_to_event',
            'minutes_since_event',
            'pre_event_proximity',
            'post_event_proximity',
            'event_impact',
            'event_category',
            'avoid_entry',
            'volatility_expected',
        ]

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                'minutes_to_event': gym.spaces.Box(0, math.inf, shape=(1,)),
                'minutes_since_event': gym.spaces.Box(0, math.inf, shape=(1,)),
                'pre_event_proximity': gym.spaces.Box(0, 1, shape=(1,)),
                'post_event_proximity': gym.spaces.Box(0, 1, shape=(1,)),
                'event_impact': gym.spaces.Box(0, 1, shape=(1,)),
                'event_category': gym.spaces.Discrete(len(EVENT_CATEGORIES)),
                'avoid_entry': gym.spaces.Discrete(2),
                'volatility_expected': gym.spaces.Box(0, 1, shape=(1,)),
            })
        else:
            self.spaces = OrderedDict()

        self._events = self._load(calendar_path)

    def _load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=['datetime'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df = df[df['currency'].str.upper().isin(self.currencies)].copy()
        df['impact'] = df['impact'].astype(int).clip(1, 3)
        df['category'] = df['event'].apply(_classify_event)
        df = df.sort_values('datetime').reset_index(drop=True)
        return df

    def reset(self):
        pass

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        time = getattr(frame, 'time_end', None) or getattr(frame, 'time_start', None)

        if time is None or self._events.empty:
            values = [9999.0, 9999.0, 0.0, 0.0, 0.0, 0, 0, 0.0]
            if self.write_to_state:
                for name, val in zip(self.names, values):
                    state[name] = val
            if self.write_to_frame:
                for name, val in zip(self.names, values):
                    setattr(frame, name, val)
            return

        dt = time.datetime
        now = pd.Timestamp(dt) if dt.tzinfo else pd.Timestamp(dt, tz='UTC')
        event_times = self._events['datetime']

        future = self._events[event_times >= now]
        past = self._events[event_times < now]

        minutes_to = 9999.0
        next_impact = 0
        next_category = 0
        if not future.empty:
            next_row = future.iloc[0]
            minutes_to = (next_row['datetime'] - now).total_seconds() / 60
            next_impact = next_row['impact']
            next_category = int(next_row['category'])

        minutes_since = 9999.0
        prev_impact = 0
        if not past.empty:
            prev_row = past.iloc[-1]
            minutes_since = (now - prev_row['datetime']).total_seconds() / 60
            prev_impact = prev_row['impact']

        impact_weight_next = IMPACT_WEIGHTS.get(next_impact, 0.0)
        impact_weight_prev = IMPACT_WEIGHTS.get(prev_impact, 0.0)

        pre_proximity = 0.0
        if minutes_to < self.pre_event_window:
            raw = math.exp(-self._decay_lambda * minutes_to)
            pre_proximity = raw * impact_weight_next

        post_proximity = 0.0
        if minutes_since < self.post_event_window:
            raw = math.exp(-self._decay_lambda * minutes_since)
            post_proximity = raw * impact_weight_prev

        event_impact = max(
            impact_weight_next if minutes_to < self.pre_event_window else 0.0,
            impact_weight_prev if minutes_since < self.post_event_window else 0.0,
        )

        avoid = int(
            (minutes_to < 15 and next_impact >= 2) or
            (minutes_since < 15 and prev_impact >= 3)
        )

        volatility = min(1.0, pre_proximity + post_proximity)

        values = [
            minutes_to,
            minutes_since,
            pre_proximity,
            post_proximity,
            event_impact,
            next_category,
            avoid,
            volatility,
        ]

        if self.write_to_state:
            for name, val in zip(self.names, values):
                state[name] = val
        if self.write_to_frame:
            for name, val in zip(self.names, values):
                setattr(frame, name, val)
