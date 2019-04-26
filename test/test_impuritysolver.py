import unittest
from pytriqs.operators import c, c_dag
from pytriqs.gf.local import BlockGf, GfImFreq, SemiCircular, iOmega_n, inverse

from cdmft.impuritysolver import ImpuritySolver


class TestImpuritySolver(unittest.TestCase):

    def test_ImpuritySolver_initialization(self):
        solver = ImpuritySolver(10, {'u': range(2)}, 1025, 10001, 50)

    def test_ImpuritySolver_run(self):
        solver = ImpuritySolver(10, {'u': range(2)}, 100, 1000, 25)
        h = c_dag('u', 0) * c('u', 0)
        g = BlockGf(name_list = ['u'],
                    block_list = [GfImFreq(indices = range(2),
                                           beta = 10, n_points = 100)])
        g['u'] << SemiCircular(1)
        solver.run(g, h, 0, n_cycles = 50, length_cycle = 2, n_warmup_cycles = 20, verbosity = 0)
        res = solver.get_results()

    def test_ImpuritySolver_init_new_giw(self):
        solver = ImpuritySolver(10, {'u': range(2)}, 100, 1000, 25)
        solver._init_new_giw()

    def test_ImpuritySolver_get_g_iw(self):
        solver = ImpuritySolver(10, {'u': range(2)}, 100, 1000, 25)
        h = c_dag('u', 0) * c('u', 0)
        g = BlockGf(name_list = ['u'],
                    block_list = [GfImFreq(indices = range(2),
                                           beta = 10, n_points = 100)])
        g['u'] << SemiCircular(1)
        solver.run(g, h, 0, n_cycles = 50, length_cycle = 2, n_warmup_cycles = 20, verbosity = 0, perform_post_proc = True, measure_g_l = True)
        g << solver.get_g_iw(True, False)
        g << solver.get_g_iw(False, True)
        g << solver.get_se(True, False)
        g << solver.get_se(False, True)
