import unittest, os

from bethe.parameters import DefaultDMFTParameters
from bethe.dmft import DMFT
from bethe.systems import SingleSiteBethe


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
