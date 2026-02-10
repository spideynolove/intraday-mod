from typing import Optional, Union, Literal
from datetime import datetime, timezone, timedelta
import requests
import pandas as pd
import arrow


_FXSTREET_URL = 'https://calendar-api.fxsstatic.com/en/api/v2/eventDates/{start}/{end}'
_FXSTREET_HEADERS = {
    'accept': 'application/json',
    'origin': 'https://www.fxstreet.com',
    'referer': 'https://www.fxstreet.com/',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/144.0.0.0 Safari/537.36',
}
_FXSTREET_PARAMS = (
    'volatilities=MEDIUM&volatilities=HIGH'
    '&countries=US&countries=UK&countries=EMU&countries=DE'
    '&countries=JP&countries=CA&countries=AU&countries=NZ&countries=CH'
)

_FXEMPIRE_URL = 'https://www.fxempire.com/api/v1/en/economic-calendar'
_FXEMPIRE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0',
    'Referer': 'https://www.fxempire.com/tools/economic-calendar',
}
_VOLATILITY_MAP = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}


def _fetch_fxstreet(date_from: datetime, date_to: datetime) -> pd.DataFrame:
    start = date_from.strftime('%Y-%m-%dT%H:%M:%SZ')
    end = date_to.strftime('%Y-%m-%dT%H:%M:%SZ')
    url = f"{_FXSTREET_URL.format(start=start, end=end)}?{_FXSTREET_PARAMS}"
    r = requests.get(url, headers=_FXSTREET_HEADERS, timeout=15)
    r.raise_for_status()
    events = r.json()
    if not events:
        return pd.DataFrame()
    rows = []
    for e in events:
        rows.append({
            'datetime': pd.Timestamp(e['dateUtc']),
            'currency': e.get('currencyCode', ''),
            'event': e.get('name', ''),
            'impact': _VOLATILITY_MAP.get(e.get('volatility', 'LOW'), 1),
            'actual': e.get('actual'),
            'forecast': e.get('consensus'),
            'previous': e.get('previous'),
            'is_better': e.get('isBetterThanExpected'),
        })
    df = pd.DataFrame(rows)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    return df


def _fetch_fxempire(date_from: datetime, date_to: datetime) -> pd.DataFrame:
    params = {
        'impact': '2,3',
        'country': 'united-states,united-kingdom,euro-area,germany,japan,canada,australia,new-zealand,switzerland',
        'dateFrom': date_from.strftime('%Y-%m-%d'),
        'dateTo': date_to.strftime('%Y-%m-%d'),
    }
    r = requests.get(_FXEMPIRE_URL, params=params, headers=_FXEMPIRE_HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()
    rows = []
    for day in data.get('calendar', []):
        for e in day.get('events', []):
            rows.append({
                'datetime': pd.Timestamp(e['date']),
                'currency': _country_to_currency(e.get('country', '')),
                'event': e.get('name', ''),
                'impact': int(e.get('impact', 1)),
                'actual': _parse_pct(e.get('actual')),
                'forecast': _parse_pct(e.get('forecast')),
                'previous': _parse_pct(e.get('previous')),
                'is_better': e.get('color') == 'above',
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    return df


def _parse_pct(value: Optional[str]) -> Optional[float]:
    if value is None or value == '':
        return None
    try:
        return float(str(value).replace('%', '').replace('K', '').replace('B', '').replace('M', '').strip())
    except (ValueError, AttributeError):
        return None


def _country_to_currency(country: str) -> str:
    mapping = {
        'united-states': 'USD', 'united-kingdom': 'GBP', 'euro-area': 'EUR',
        'germany': 'EUR', 'japan': 'JPY', 'canada': 'CAD', 'australia': 'AUD',
        'new-zealand': 'NZD', 'switzerland': 'CHF', 'china': 'CNY',
    }
    return mapping.get(country.lower(), '')


def fetch_calendar(
    date_from: datetime,
    date_to: datetime,
    source: Literal['fxstreet', 'fxempire', 'auto'] = 'auto',
) -> pd.DataFrame:
    if source == 'fxstreet':
        return _fetch_fxstreet(date_from, date_to)
    if source == 'fxempire':
        return _fetch_fxempire(date_from, date_to)
    try:
        df = _fetch_fxstreet(date_from, date_to)
        if not df.empty:
            return df
    except Exception:
        pass
    return _fetch_fxempire(date_from, date_to)


def build_calendar_csv(
    date_from: datetime,
    date_to: datetime,
    output_path: str,
    chunk_days: int = 30,
    source: Literal['fxstreet', 'fxempire', 'auto'] = 'auto',
):
    all_dfs = []
    current = date_from
    while current < date_to:
        chunk_end = min(current + timedelta(days=chunk_days), date_to)
        df = fetch_calendar(current, chunk_end, source=source)
        if not df.empty:
            all_dfs.append(df)
        current = chunk_end + timedelta(seconds=1)
    if all_dfs:
        result = pd.concat(all_dfs).drop_duplicates(subset=['datetime', 'event', 'currency'])
        result = result.sort_values('datetime').reset_index(drop=True)
        result.to_csv(output_path, index=False)
        return result
    return pd.DataFrame()
