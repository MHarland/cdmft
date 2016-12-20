import unittest, os, numpy as np

from bethe.parameters import TestDMFTParameters
from bethe.selfconsistency import Cycle
from bethe.storage import LoopStorage
from bethe.schemes.bethe import GLocal, SelfEnergy, WeissField
from bethe.operators.hubbard import Site


class TestCycle(unittest.TestCase):

    def test_Cycle_initialization(self):
        sto = LoopStorage("test.h5")
        params = TestDMFTParameters()
        h = Site(2)
        gloc = GLocal(1, {'up': np.array([[0]]), 'dn': np.array([[0]])}, ['up', 'dn'], [1, 1], 10, 1025)
        g0 = WeissField(['up', 'dn'], [1, 1], 10, 1025)
        se = SelfEnergy(['up', 'dn'], [1, 1], 10, 1025)
        mu = 0
        cyc = Cycle(sto, params, h.get_h_int(), gloc, g0, se, mu)
        os.remove("test.h5")

    def test_Cycle_run(self):
        sto = LoopStorage("test.h5")
        params = TestDMFTParameters()
        h = Site(2)
        gloc = GLocal(1, {'up': np.array([[0]]), 'dn': np.array([[0]])}, ['up', 'dn'], [1, 1], 10, 1025)
        g0 = WeissField(['up', 'dn'], [1, 1], 10, 1025)
        se = SelfEnergy(['up', 'dn'], [1, 1], 10, 1025)
        mu = 1
        cyc = Cycle(sto, params, h.get_h_int(), gloc, g0, se, mu)
        cyc.run(3)
        os.remove("test.h5")
