import unittest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import pandas as pd
from intraday.providers.calendar_provider import (
    fetch_calendar, _fetch_fxstreet, _fetch_fxempire,
    _parse_pct, _country_to_currency,
)

_FXSTREET_SAMPLE = [
    {'dateUtc': '2026-02-10T13:30:00Z', 'name': 'NFP', 'currencyCode': 'USD',
     'volatility': 'HIGH', 'actual': 215.0, 'consensus': 200.0, 'previous': 199.0,
     'isBetterThanExpected': True},
    {'dateUtc': '2026-02-12T12:00:00Z', 'name': 'CPI MoM', 'currencyCode': 'USD',
     'volatility': 'HIGH', 'actual': 0.3, 'consensus': 0.2, 'previous': 0.1,
     'isBetterThanExpected': True},
]

_FXEMPIRE_SAMPLE = {'calendar': [{'day': '2026-02-10', 'formattedDay': {}, 'events': [
    {'id': 1, 'date': '2026-02-10T13:30:00.000Z', 'name': 'NFP', 'impact': 3,
     'country': 'united-states', 'actual': '215K', 'forecast': '200K', 'previous': '199K', 'color': 'above'},
]}]}


class TestCalendarProvider(unittest.TestCase):
    def _make_response(self, data):
        m = MagicMock()
        m.raise_for_status.return_value = None
        m.json.return_value = data
        return m

    def test_parse_pct_values(self):
        self.assertAlmostEqual(_parse_pct('1.2%'), 1.2)
        self.assertAlmostEqual(_parse_pct('215K'), 215.0)
        self.assertIsNone(_parse_pct(''))
        self.assertIsNone(_parse_pct(None))

    def test_country_to_currency(self):
        self.assertEqual(_country_to_currency('united-states'), 'USD')
        self.assertEqual(_country_to_currency('euro-area'), 'EUR')
        self.assertEqual(_country_to_currency('unknown'), '')

    @patch('intraday.providers.calendar_provider.requests.get')
    def test_fetch_fxstreet(self, mock_get):
        mock_get.return_value = self._make_response(_FXSTREET_SAMPLE)
        df = _fetch_fxstreet(datetime(2026, 2, 10, tzinfo=timezone.utc), datetime(2026, 2, 14, tzinfo=timezone.utc))
        self.assertEqual(len(df), 2)
        self.assertIn('datetime', df.columns)
        self.assertIn('impact', df.columns)
        self.assertEqual(df.iloc[0]['currency'], 'USD')
        self.assertEqual(df.iloc[0]['impact'], 3)

    @patch('intraday.providers.calendar_provider.requests.get')
    def test_fetch_fxempire(self, mock_get):
        mock_get.return_value = self._make_response(_FXEMPIRE_SAMPLE)
        df = _fetch_fxempire(datetime(2026, 2, 10, tzinfo=timezone.utc), datetime(2026, 2, 14, tzinfo=timezone.utc))
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['currency'], 'USD')
        self.assertEqual(df.iloc[0]['impact'], 3)
        self.assertTrue(df.iloc[0]['is_better'])

    @patch('intraday.providers.calendar_provider.requests.get')
    def test_auto_fallback_to_fxempire(self, mock_get):
        mock_get.side_effect = [Exception('fxstreet down'), self._make_response(_FXEMPIRE_SAMPLE)]
        df = fetch_calendar(
            datetime(2026, 2, 10, tzinfo=timezone.utc),
            datetime(2026, 2, 14, tzinfo=timezone.utc),
            source='auto'
        )
        self.assertEqual(len(df), 1)

    @patch('intraday.providers.calendar_provider.requests.get')
    def test_fxstreet_empty_returns_dataframe(self, mock_get):
        mock_get.return_value = self._make_response([])
        df = _fetch_fxstreet(datetime(2026, 2, 10, tzinfo=timezone.utc), datetime(2026, 2, 14, tzinfo=timezone.utc))
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)


if __name__ == '__main__':
    unittest.main()
