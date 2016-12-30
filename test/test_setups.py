import unittest, os, numpy as np

from bethe.h5interface import Storage
from bethe.selfconsistency import Cycle
from bethe.setups.bethelattice import SingleBetheSetup, TriangleBetheSetup, PlaquetteBetheSetup
from bethe.setups.cdmftchain import MomentumDimerSetup, StrelSetup
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

    def test_chain_StrelCDMFTSetup(self):
        setup = StrelSetup(10, 0, -.1, -1, -.1, 0, 2, 16)
        sto = Storage('test.h5')
        par = TestDMFTParameters()
        cyc = Cycle(sto, par, **setup.initialize_cycle())
        cyc.run(1, n_cycles = 0)
        setup = StrelSetup(10, 0, -1, -1, -1, 1, 2, 8)
        setup.transform_sites(np.pi/4., np.pi/4.)
        e_loc =  np.sum([w * e['up-d'] for k,w,e in setup.gloc.lat.loop_over_bz()], axis = 0)
        self.assertTrue(np.allclose(e_loc[0, 1], 0))
        setup.transform_sites(2.657, 9.52)
        e_loc =  np.sum([w * e['up-d'] for k,w,e in setup.gloc.lat.loop_over_bz()], axis = 0)
        self.assertTrue(not np.allclose(e_loc[0, 1], 0))
        setup.transform_sites(np.pi/4., np.pi/4.)
        e_loc =  np.sum([w * e['up-d'] for k,w,e in setup.gloc.lat.loop_over_bz()], axis = 0)
        self.assertTrue(np.allclose(e_loc[0, 1], 0))
        cyc = Cycle(sto, par, **setup.initialize_cycle())
        cyc.run(1, n_cycles = 0)
        os.remove('test.h5')
