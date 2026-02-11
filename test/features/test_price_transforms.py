import unittest
from collections import OrderedDict
from intraday.features.price_transforms import AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE
from intraday.frame import Frame


class TestAVGPRICE(unittest.TestCase):
    def test_avgprice_initializes(self):
        avgprice = AVGPRICE(write_to='state')
        self.assertEqual(avgprice.names, ['avgprice'])

    def test_avgprice_calculation(self):
        avgprice = AVGPRICE(write_to='state')
        avgprice.reset()

        frames = [Frame(open=100, high=110, low=90, close=105)]
        state = OrderedDict()
        avgprice.process(frames, state)

        expected = (100 + 110 + 90 + 105) / 4
        self.assertAlmostEqual(state['avgprice'], expected, places=2)


class TestMEDPRICE(unittest.TestCase):
    def test_medprice_initializes(self):
        medprice = MEDPRICE(write_to='state')
        self.assertEqual(medprice.names, ['medprice'])

    def test_medprice_calculation(self):
        medprice = MEDPRICE(write_to='state')
        medprice.reset()

        frames = [Frame(high=110, low=90)]
        state = OrderedDict()
        medprice.process(frames, state)

        expected = (110 + 90) / 2
        self.assertAlmostEqual(state['medprice'], expected, places=2)


class TestTYPPRICE(unittest.TestCase):
    def test_typprice_initializes(self):
        typprice = TYPPRICE(write_to='state')
        self.assertEqual(typprice.names, ['typprice'])

    def test_typprice_calculation(self):
        typprice = TYPPRICE(write_to='state')
        typprice.reset()

        frames = [Frame(high=110, low=90, close=105)]
        state = OrderedDict()
        typprice.process(frames, state)

        expected = (110 + 90 + 105) / 3
        self.assertAlmostEqual(state['typprice'], expected, places=2)


class TestWCLPRICE(unittest.TestCase):
    def test_wclprice_initializes(self):
        wclprice = WCLPRICE(write_to='state')
        self.assertEqual(wclprice.names, ['wclprice'])

    def test_wclprice_calculation(self):
        wclprice = WCLPRICE(write_to='state')
        wclprice.reset()

        frames = [Frame(high=110, low=90, close=105)]
        state = OrderedDict()
        wclprice.process(frames, state)

        expected = (110 + 90 + 2 * 105) / 4
        self.assertAlmostEqual(state['wclprice'], expected, places=2)


if __name__ == '__main__':
    unittest.main()
