import unittest
from collections import OrderedDict
import arrow
from intraday.features import SwingStructure, PriceZones, OrderBlock, LiquiditySweep, SessionLevels
from intraday.frame import Frame


def make_frame(high, low, close, open_=None, volume=1000, hour=10):
    f = Frame(
        high=high, low=low, close=close,
        open=open_ or close - 0.5,
        volume=volume,
    )
    f.buy_volume = volume * 0.55
    f.sell_volume = volume * 0.45
    t = arrow.Arrow(2024, 1, 15, hour, 0, 0)
    f.time_start = t
    f.time_end = t.shift(hours=1)
    return f


class TestSMCIntegration(unittest.TestCase):
    def test_all_features_produce_outputs(self):
        features = [
            SwingStructure(swing_period=2),
            PriceZones(range_period=5),
            OrderBlock(impulse_threshold=1.5),
            LiquiditySweep(swing_period=2),
            SessionLevels(),
        ]
        for f in features:
            f.reset()

        frames = []
        for i in range(15):
            h = 100 + i + (i % 3)
            l = 98 + i - (i % 2)
            c = 99 + i
            frames.append(make_frame(h, l, c, hour=8 + (i % 8)))

        state = OrderedDict()
        for i in range(len(frames)):
            state = OrderedDict()
            for feature in features:
                feature.process(frames[:i+1], state)

        expected_keys = [
            'swing_high_detected', 'bos_bullish',
            'zone_type', 'equilibrium_price',
            'bullish_ob_detected', 'fvg_bullish_detected',
            'liquidity_above', 'sweep_high_detected',
            'session_type', 'in_kill_zone',
        ]
        for key in expected_keys:
            self.assertIn(key, state, f"Missing key: {key}")

    def test_no_name_collisions(self):
        features = [
            SwingStructure(), PriceZones(), OrderBlock(),
            LiquiditySweep(), SessionLevels(),
        ]
        all_names = []
        for f in features:
            all_names.extend(f.names)
        self.assertEqual(len(all_names), len(set(all_names)), "Name collision detected")


if __name__ == '__main__':
    unittest.main()
