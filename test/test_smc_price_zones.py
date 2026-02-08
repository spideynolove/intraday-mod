import unittest
from collections import OrderedDict
from intraday.features.smc_price_zones import PriceZones
from intraday.frame import Frame


class TestPriceZones(unittest.TestCase):
    def test_initializes(self):
        p = PriceZones(range_period=10)
        self.assertIn('zone_type', p.names)
        self.assertIn('equilibrium_price', p.names)
        self.assertIn('displacement_bullish', p.names)
        self.assertIn('volume_imbalance_ratio', p.names)

    def test_premium_zone(self):
        p = PriceZones(range_period=3)
        p.reset()
        frames = [
            Frame(high=100, low=90, close=95, open=93, buy_volume=500, sell_volume=500),
            Frame(high=105, low=95, close=100, open=96, buy_volume=500, sell_volume=500),
            Frame(high=110, low=100, close=108, open=101, buy_volume=500, sell_volume=500),
        ]
        for i in range(len(frames)):
            state = OrderedDict()
            p.process(frames[:i+1], state)
        self.assertEqual(state['zone_type'], 1)

    def test_discount_zone(self):
        p = PriceZones(range_period=3)
        p.reset()
        frames = [
            Frame(high=110, low=100, close=105, open=103, buy_volume=500, sell_volume=500),
            Frame(high=105, low=95, close=100, open=104, buy_volume=500, sell_volume=500),
            Frame(high=100, low=90, close=92, open=99, buy_volume=500, sell_volume=500),
        ]
        for i in range(len(frames)):
            state = OrderedDict()
            p.process(frames[:i+1], state)
        self.assertEqual(state['zone_type'], 2)

    def test_displacement_detection(self):
        p = PriceZones(range_period=3, displacement_threshold=0.5)
        p.reset()
        frames = []
        for i in range(10):
            frames.append(Frame(high=100+i, low=99+i, close=100+i, open=99+i,
                                buy_volume=500, sell_volume=500))
        frames.append(Frame(high=130, low=109, close=129, open=110,
                           buy_volume=2000, sell_volume=200))
        for i in range(len(frames)):
            state = OrderedDict()
            p.process(frames[:i+1], state)
        self.assertEqual(state['displacement_bullish'], 1)

    def test_volume_imbalance(self):
        p = PriceZones(range_period=3)
        p.reset()
        frames = [
            Frame(high=100, low=95, close=98, open=96, buy_volume=900, sell_volume=100),
        ]
        state = OrderedDict()
        p.process(frames, state)
        self.assertGreater(state['volume_imbalance_ratio'], 1.0)

    def test_reset_clears_state(self):
        p = PriceZones(range_period=3)
        frames = [Frame(high=110, low=90, close=100, open=95,
                       buy_volume=500, sell_volume=500)]
        p.process(frames, OrderedDict())
        p.reset()
        state = OrderedDict()
        p.process([Frame(high=100, low=95, close=97, open=96,
                        buy_volume=500, sell_volume=500)], state)
        self.assertIn('zone_type', state)


if __name__ == '__main__':
    unittest.main()
