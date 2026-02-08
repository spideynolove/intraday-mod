import unittest
from collections import OrderedDict
import arrow
from intraday.features.smc_session_levels import SessionLevels
from intraday.frame import Frame


def make_frame(hour_utc: int, high: float, low: float, close: float) -> Frame:
    t = arrow.Arrow(2024, 1, 15, hour_utc, 0, 0)
    f = Frame(high=high, low=low, close=close)
    f.time_start = t
    f.time_end = t.shift(hours=1)
    return f


class TestSessionLevels(unittest.TestCase):
    def test_initializes(self):
        sl = SessionLevels()
        self.assertIn('session_type', sl.names)
        self.assertIn('asian_high', sl.names)
        self.assertIn('in_kill_zone', sl.names)

    def test_asian_session(self):
        sl = SessionLevels()
        sl.reset()
        frames = [make_frame(2, 100, 90, 95)]
        state = OrderedDict()
        sl.process(frames, state)
        self.assertEqual(state['session_type'], 1)
        self.assertAlmostEqual(state['asian_high'], 100.0)

    def test_london_session(self):
        sl = SessionLevels()
        sl.reset()
        frames = [make_frame(9, 105, 95, 100)]
        state = OrderedDict()
        sl.process(frames, state)
        self.assertEqual(state['session_type'], 2)

    def test_ny_session(self):
        sl = SessionLevels()
        sl.reset()
        frames = [make_frame(14, 110, 100, 105)]
        state = OrderedDict()
        sl.process(frames, state)
        self.assertEqual(state['session_type'], 3)

    def test_kill_zone(self):
        sl = SessionLevels()
        sl.reset()
        frames = [make_frame(8, 105, 95, 100)]
        state = OrderedDict()
        sl.process(frames, state)
        self.assertEqual(state['in_kill_zone'], 1)

    def test_session_high_low_tracking(self):
        sl = SessionLevels()
        sl.reset()
        frames = [
            make_frame(2, 100, 90, 95),
            make_frame(3, 105, 92, 103),
            make_frame(4, 103, 88, 91),
        ]
        for i in range(len(frames)):
            state = OrderedDict()
            sl.process(frames[:i+1], state)
        self.assertAlmostEqual(state['asian_high'], 105.0)
        self.assertAlmostEqual(state['asian_low'], 88.0)

    def test_reset_clears_state(self):
        sl = SessionLevels()
        frames = [make_frame(2, 100, 90, 95)]
        sl.process(frames, OrderedDict())
        sl.reset()
        state = OrderedDict()
        sl.process([make_frame(8, 105, 95, 100)], state)
        self.assertEqual(state['session_type'], 2)


if __name__ == '__main__':
    unittest.main()
