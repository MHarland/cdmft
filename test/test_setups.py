import unittest, os

from bethe.h5interface import Storage
from bethe.selfconsistency import Cycle
from bethe.setups.bethelattice import SingleBetheSetup, TriangleBetheSetup, PlaquetteBetheSetup
from bethe.setups.cdmftchain import MomentumDimerSetup
from bethe.setups.cdmftsquarelattice import MomentumPlaquetteSetup
from bethe.parameters import TestDMFTParameters


class TestSetups(unittest.TestCase):

    def test_BetheSetups_init(self):
        setup = SingleBetheSetup(10, 1, 2, 1)
        setup = TriangleBetheSetup(10, 1, 2, -1, 1)
        setup = PlaquetteBetheSetup(10, 1, 2, -1, .3, .2)
        setup.init_noninteracting()
        setup.init_centered_semicirculars()

    def test_SingleBetheSetup_with_cycle_run(self):
        setup = SingleBetheSetup(10, 1, 2, 1)
        sto = Storage('test.h5')
        par = TestDMFTParameters()
        cyc = Cycle(sto, par, **setup.initialize_cycle())
        cyc.run(1)
        setup2 = SingleBetheSetup(10, 1, 2, 1)
        setup2.set_data(sto)
        cyc = Cycle(sto, par, **setup.initialize_cycle())
        cyc.run(1)
        self.assertTrue(sto.get_completed_loops() == 2)
        self.assertTrue(.9 < sto.load("density") < 1.1)
        self.assertTrue(.99 < sto.load("average_sign"))
        os.remove('test.h5')
        
    def test_chain_MomentumDimerCDMFTSetup(self):
        setup = MomentumDimerSetup(10, 2, 4, -1, 10)
        sto = Storage('test.h5')
        par = TestDMFTParameters()
        cyc = Cycle(sto, par, **setup.initialize_cycle())
        cyc.run(1, n_cycles = 0)
        os.remove('test.h5')
        
    def test_squarelattice_MomentumPlaquetteCDMFTSetup(self):
        setup = MomentumPlaquetteSetup(10, 2, 4, -1, .3, 10)
        sto = Storage('test.h5')
        par = TestDMFTParameters()
        cyc = Cycle(sto, par, **setup.initialize_cycle())
        cyc.run(1, n_cycles = 0)
        os.remove('test.h5')
