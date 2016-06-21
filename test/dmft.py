import unittest, os

from Bethe.parameters import DefaultDMFTParameters
from Bethe.dmft import DMFT
from Bethe.systems import SingleSiteBethe


class TestDMFT(unittest.TestCase):

    def test_dmft_initialization(self):
        params = DefaultDMFTParameters()
        sys = SingleSiteBethe(10, 1, .5)
        dmft = DMFT(params, sys, "test.h5")
        os.remove("test.h5")

    def test_dmft_default_run(self):
        params = DefaultDMFTParameters()
        sys = SingleSiteBethe(10, 1, .5)
        dmft = DMFT(params, sys, "test.h5")
        dmft.run_loops(1)
        os.remove("test.h5")
