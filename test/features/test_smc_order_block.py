import unittest
from collections import OrderedDict
from intraday.features.smc_order_block import OrderBlock
from intraday.frame import Frame


class TestOrderBlock(unittest.TestCase):
    def test_initializes(self):
        ob = OrderBlock()
        self.assertIn('bullish_ob_detected', ob.names)
        self.assertIn('in_bullish_ob', ob.names)
        self.assertIn('fvg_bullish_detected', ob.names)

    def test_bullish_ob_detection(self):
        ob = OrderBlock(impulse_threshold=1.5)
        ob.reset()
        frames = [
            Frame(open=105, high=106, low=98, close=99),
            Frame(open=99, high=115, low=98, close=114),
            Frame(open=114, high=116, low=112, close=113),
        ]
        detected = False
        for i in range(len(frames)):
            state = OrderedDict()
            ob.process(frames[:i+1], state)
            if state['bullish_ob_detected'] == 1:
                detected = True
        self.assertTrue(detected)

    def test_bearish_ob_detection(self):
        ob = OrderBlock(impulse_threshold=1.5)
        ob.reset()
        frames = [
            Frame(open=100, high=108, low=99, close=107),
            Frame(open=107, high=108, low=92, close=93),
            Frame(open=93, high=95, low=91, close=92),
        ]
        detected = False
        for i in range(len(frames)):
            state = OrderedDict()
            ob.process(frames[:i+1], state)
            if state['bearish_ob_detected'] == 1:
                detected = True
        self.assertTrue(detected)

    def test_fvg_bullish_detection(self):
        ob = OrderBlock()
        ob.reset()
        frames = [
            Frame(open=100, high=102, low=98, close=101),
            Frame(open=103, high=108, low=102, close=107),
            Frame(open=107, high=115, low=105, close=113),
        ]
        for i in range(len(frames)):
            state = OrderedDict()
            ob.process(frames[:i+1], state)
        self.assertEqual(state['fvg_bullish_detected'], 1)

    def test_reset_clears_zones(self):
        ob = OrderBlock()
        frames = [Frame(open=100, high=102, low=98, close=101)]
        ob.process(frames, OrderedDict())
        ob.reset()
        state = OrderedDict()
        ob.process([Frame(open=100, high=102, low=98, close=101)], state)
        self.assertEqual(state['bullish_ob_detected'], 0)


if __name__ == '__main__':
    unittest.main()
