import unittest, os, numpy as np

from bethe.convergence import Criterion
from bethe.h5interface import Storage
from bethe.selfconsistency import Cycle
from bethe.setups.bethelattice import SingleBetheSetup
from bethe.parameters import TestDMFTParameters


class TestConvergence(unittest.TestCase):

    def test_Criterion_init(self):
        setup = SingleBetheSetup(10, 1, 2, 1)
        sto = Storage('test.h5')
        par = TestDMFTParameters(n_cycles = 500)
        cyc = Cycle(sto, par, **setup.initialize_cycle())
        cyc.run(1)
        crit = Criterion(sto, verbose = False)
        self.assertFalse(crit.confirms_convergence())
        os.remove('test.h5')

    def test_Criterion_applied(self):
        setup = SingleBetheSetup(10, 1, 2, 1)
        sto = Storage('test.h5')
        par = TestDMFTParameters(n_cycles = 300000, verbosity = 1, length_cycle = 50)
        cyc = Cycle(sto, par, **setup.initialize_cycle())
        crit = Criterion(sto, verbose = False)
        cyc.add_convergence_criterion(crit)
        cyc.run(20)
        self.assertTrue(sto.get_completed_loops() < 20)
        os.remove('test.h5')

    def test_Criterion_applied_noisy_case(self):
        setup = SingleBetheSetup(10, 1, 2, 1)
        sto = Storage('test.h5')
        par = TestDMFTParameters(n_cycles = 300000, verbosity = 1, length_cycle = 50)
        cyc = Cycle(sto, par, **setup.initialize_cycle())
        crit = Criterion(sto, lim_absgdiff = 1e-10, verbose = False)
        cyc.add_convergence_criterion(crit)
        cyc.run(20)
        self.assertTrue(sto.get_completed_loops() < 20)
        os.remove('test.h5')
