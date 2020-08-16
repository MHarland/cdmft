import unittest
import os
import numpy as np

from cdmft.h5interface import Storage
from cdmft.selfconsistency import Cycle
from cdmft.setups.bethelattice import SingleBetheSetup, TriangleBetheSetup, PlaquetteBetheSetup, TriangleAIAOBetheSetup, TwoOrbitalDimerBetheSetup, TwoOrbitalMomentumDimerBetheSetup, NambuMomentumPlaquette, AFMNambuMomentumPlaquette, TwoMixedOrbitalDimerBetheSetup
from cdmft.setups.cdmftchain import MomentumDimerSetup, StrelSetup, SingleSiteSetup
from cdmft.setups.cdmftsquarelattice import MomentumPlaquetteSetup, NambuMomentumPlaquetteSetup
from cdmft.setups.pcdmftchain import DimerChainSetup
from cdmft.parameters import TestDMFTParameters


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

    def test_chain_SingleSiteCDMFTSetup(self):
        setup = SingleSiteSetup(10, 2, 4, -1, 10)
        sto = Storage('test.h5')
        par = TestDMFTParameters(n_cycles=0)
        cyc = Cycle(sto, par, **setup.initialize_cycle())
        cyc.run(1)
        os.remove('test.h5')

    def test_chain_MomentumDimerCDMFTSetup(self):
        setup = MomentumDimerSetup(10, 2, 4, -1, 10)
        sto = Storage('test.h5')
        par = TestDMFTParameters(n_cycles=0)
        cyc = Cycle(sto, par, **setup.initialize_cycle())
        cyc.run(1)
        os.remove('test.h5')

    def test_squarelattice_MomentumPlaquetteCDMFTSetup(self):
        setup = MomentumPlaquetteSetup(10, 2, 4, -1, .3, 10)
        sto = Storage('test.h5')
        par = TestDMFTParameters(n_cycles=0)
        cyc = Cycle(sto, par, **setup.initialize_cycle())
        cyc.run(1)
        os.remove('test.h5')

    def test_squarelattice_PlaquetteCDMFT_hoppingsymmetry(self):
        for tb in ['2d', '3d', '3dandersen']:
            setup = MomentumPlaquetteSetup(
                10, 2, 4, -1, .3, 16, tightbinding=tb)
            hloc = np.sum(setup.disp.energies_k, axis=0) / \
                setup.disp.energies_k.shape[0]
            for i in range(3):
                self.assertAlmostEqual(hloc[i, i], hloc[i+1, i+1])
            for i, j in [(0, 1), (0, 2), (1, 3), (2, 3), (2, 0), (3, 1), (3, 2)]:
                self.assertAlmostEqual(hloc[1, 0], hloc[i, j])
            for i, j in [(0, 3), (1, 2), (2, 1)]:
                self.assertAlmostEqual(hloc[3, 0], hloc[i, j])

    def test_squarelattice_NambuMomentumPlaquetteCDMFTSetup(self):
        setup = NambuMomentumPlaquetteSetup(10, 2, 4, -1, .3, 10)
        sto = Storage('test.h5')
        par = TestDMFTParameters(n_cycles=0)
        cyc = Cycle(sto, par, **setup.initialize_cycle())
        cyc.run(1)
        os.remove('test.h5')

    def test_chain_StrelCDMFTSetup(self):
        setup = StrelSetup(10, 0, -.1, -1, -.1, 0, 2, 16)
        sto = Storage('test.h5')
        par = TestDMFTParameters(n_cycles=0)
        cyc = Cycle(sto, par, **setup.initialize_cycle())
        cyc.run(1)
        setup = StrelSetup(10, 0, -1, -1, -1, 1, 2, 8)
        setup.transform_sites(np.pi/4., np.pi/4.)
        e_loc = np.sum([w * e['up-d']
                        for k, w, e in setup.gloc.lat.loop_over_bz()], axis=0)
        self.assertTrue(np.allclose(e_loc[0, 1], 0))
        setup.transform_sites(2.657, 9.52)
        e_loc = np.sum([w * e['up-d']
                        for k, w, e in setup.gloc.lat.loop_over_bz()], axis=0)
        self.assertTrue(not np.allclose(e_loc[0, 1], 0))
        setup.transform_sites(np.pi/4., np.pi/4.)
        e_loc = np.sum([w * e['up-d']
                        for k, w, e in setup.gloc.lat.loop_over_bz()], axis=0)
        self.assertTrue(np.allclose(e_loc[0, 1], 0))
        cyc = Cycle(sto, par, **setup.initialize_cycle())
        cyc.run(1)
        os.remove('test.h5')
        setup = StrelSetup(10, 0, -.1, -1, -.1, 0, 2, 16)

    def test_TriangleAIAOBetheSetup(self):
        setup = TriangleAIAOBetheSetup(10, 1, 2, -1, 1)
        setup_paramag = TriangleBetheSetup(10, 1, 2, -1, 1)
        se0, g00 = setup_paramag.se, setup_paramag.g0
        setup.set_initial_guess(se0, g00, .15, -.15, .15)
        zeros = [(0, 1), (0, 3), (0, 5), (1, 2), (1, 4),
                 (2, 3), (2, 5), (3, 4), (4, 5)]
        for zero in zeros:
            i, j = zero
            #self.assertTrue(np.allclose(setup.se['spin-mom'].data[1025,i,j], 0.))
            #self.assertTrue(np.allclose(setup.se['spin-mom'].data[1025,j,i], 0.))

    def test_TwoOrbitalDimerBetheSetup(self):
        setup = TwoOrbitalDimerBetheSetup(
            10, .5, 1, .2, -1, -.1, .2, .2, .2, .2)

    def test_TwoOrbitalMomentumDimerBetheSetup(self):
        setup = TwoOrbitalMomentumDimerBetheSetup(
            10, .5, 1, .2, -1, -.1, 1, .1)

    def test_TwoMixedOrbitalMomentumDimerBetheSetup(self):
        setup = TwoMixedOrbitalDimerBetheSetup(
            10, .5, 1, .2, -1, -.1, .2, .2, .1)
        sto = Storage('test.h5')
        par = TestDMFTParameters(n_cycles=0, filling=None)
        cyc = Cycle(sto, par, **setup.initialize_cycle())
        cyc.run(1)
        os.remove('test.h5')

    def test_NambuMomentumPlaquetteSetup(self):
        setup = NambuMomentumPlaquette(100, 0, 3, -1, 0, 1)
        sto = Storage('test.h5')
        par = TestDMFTParameters(n_cycles=0, filling=None)
        cyc = Cycle(sto, par, **setup.initialize_cycle())
        cyc.run(1)
        os.remove('test.h5')
        self.assertAlmostEqual(cyc.g_loc.total_density_nambu().real, 4.)
        setup.apply_dynamical_sc_field(.2)
        setup2 = PlaquetteBetheSetup(100, 0, 3, -1, 0, 1)
        setup.transform_to_nambu(setup2.gloc, setup.gloc)

    def test_AFMNambuMomentumPlaquetteSetup(self):
        setup = AFMNambuMomentumPlaquette(100, 1, 3, -1, .3, 1)
        sto = Storage('test.h5')
        par = TestDMFTParameters(n_cycles=0, filling=None)
        cyc = Cycle(sto, par, **setup.initialize_cycle())
        cyc.run(1)
        os.remove('test.h5')
        setup.add_staggered_field(.2)
        setup.apply_dynamical_sc_field(.2)
        tbg, tbm, tbx, tby, tbgm, tbxy = .1, .2, .3, .4, .5, .6
        tb = {'GM': np.array([[tbg, 0, tbgm, 0],
                              [0, -tbg, 0, -tbgm],
                              [tbgm, 0, tbm, 0],
                              [0, -tbgm, 0, -tbm]]),
              'XY': np.array([[tbx, 0, tbxy, 0],
                              [0, -tbx, 0, -tbxy],
                              [tbxy, 0, tby, 0],
                              [0, -tbxy, 0, -tby]])}
        setup = AFMNambuMomentumPlaquette(100, 1, 3, -1, .3, tb)
        sto = Storage('test.h5')
        par = TestDMFTParameters(n_cycles=0, filling=None)
        cyc = Cycle(sto, par, **setup.initialize_cycle())
        cyc.run(1)
        os.remove('test.h5')

    def test_PCDMFTSetup(self):
        setup = DimerChainSetup(10, 1, 2, 8)
