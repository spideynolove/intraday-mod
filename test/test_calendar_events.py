import unittest
import tempfile
import os
import csv
from collections import OrderedDict
import arrow
from intraday.features.calendar_events import CalendarEvents
from intraday.frame import Frame


def make_calendar(path: str, events: list[tuple]):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['datetime', 'currency', 'event', 'impact'])
        for row in events:
            writer.writerow(row)


def make_frame(hour: int) -> Frame:
    t = arrow.Arrow(2024, 1, 15, hour, 0, 0)
    f = Frame(high=1.0855, low=1.0830, close=1.0845)
    f.time_start = t
    f.time_end = t.shift(hours=1)
    return f


class TestCalendarEvents(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cal_path = os.path.join(self.temp_dir, 'calendar.csv')
        make_calendar(self.cal_path, [
            ('2024-01-15 14:30:00+00:00', 'USD', 'Non-Farm Payroll', 3),
            ('2024-01-15 19:00:00+00:00', 'USD', 'FOMC Rate Decision', 3),
            ('2024-01-16 13:30:00+00:00', 'USD', 'CPI m/m', 2),
        ])

    def test_initializes(self):
        feat = CalendarEvents(self.cal_path, currencies=['USD'])
        self.assertIn('minutes_to_event', feat.names)
        self.assertIn('pre_event_proximity', feat.names)
        self.assertIn('avoid_entry', feat.names)

    def test_far_from_event(self):
        feat = CalendarEvents(self.cal_path, currencies=['USD'], pre_event_window=60)
        frames = [make_frame(8)]
        state = OrderedDict()
        feat.process(frames, state)
        self.assertGreater(state['minutes_to_event'], 300)
        self.assertAlmostEqual(state['pre_event_proximity'], 0.0)
        self.assertEqual(state['avoid_entry'], 0)

    def test_close_to_event_has_proximity(self):
        feat = CalendarEvents(self.cal_path, currencies=['USD'], pre_event_window=60)
        frame = Frame(high=1.0855, low=1.0830, close=1.0845)
        frame.time_start = arrow.Arrow(2024, 1, 15, 14, 15, 0)
        frame.time_end = arrow.Arrow(2024, 1, 15, 14, 15, 0)
        state = OrderedDict()
        feat.process([frame], state)
        self.assertGreater(state['pre_event_proximity'], 0.0)
        self.assertLessEqual(state['pre_event_proximity'], 1.0)

    def test_avoid_entry_before_high_impact(self):
        feat = CalendarEvents(self.cal_path, currencies=['USD'], pre_event_window=60)
        frame = Frame(high=1.0855, low=1.0830, close=1.0845)
        frame.time_start = arrow.Arrow(2024, 1, 15, 14, 20, 0)
        frame.time_end = arrow.Arrow(2024, 1, 15, 14, 20, 0)
        state = OrderedDict()
        feat.process([frame], state)
        self.assertEqual(state['avoid_entry'], 1)

    def test_currency_filter(self):
        feat = CalendarEvents(self.cal_path, currencies=['GBP'])
        self.assertTrue(feat._events.empty)
        frames = [make_frame(14)]
        state = OrderedDict()
        feat.process(frames, state)
        self.assertAlmostEqual(state['pre_event_proximity'], 0.0)

    def test_post_event_proximity(self):
        feat = CalendarEvents(self.cal_path, currencies=['USD'], post_event_window=30)
        frame = Frame(high=1.0855, low=1.0830, close=1.0845)
        frame.time_start = arrow.Arrow(2024, 1, 15, 14, 40, 0)
        frame.time_end = arrow.Arrow(2024, 1, 15, 14, 40, 0)
        state = OrderedDict()
        feat.process([frame], state)
        self.assertGreater(state['post_event_proximity'], 0.0)

    def test_event_category_detected(self):
        feat = CalendarEvents(self.cal_path, currencies=['USD'])
        frame = Frame(high=1.0855, low=1.0830, close=1.0845)
        frame.time_start = arrow.Arrow(2024, 1, 15, 14, 25, 0)
        frame.time_end = arrow.Arrow(2024, 1, 15, 14, 25, 0)
        state = OrderedDict()
        feat.process([frame], state)
        self.assertEqual(state['event_category'], 1)

    def test_reset_no_op(self):
        feat = CalendarEvents(self.cal_path, currencies=['USD'])
        feat.reset()
        self.assertFalse(feat._events.empty)


if __name__ == '__main__':
    unittest.main()
