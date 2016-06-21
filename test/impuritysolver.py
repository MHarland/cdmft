import unittest
from pytriqs.operators import c, c_dag
from pytriqs.gf.local import BlockGf, GfImFreq, SemiCircular, iOmega_n, inverse

from Bethe.impuritysolver import ImpuritySolver


class TestImpuritySolver(unittest.TestCase):

    def test_ImpuritySolver_initialization(self):
        solver = ImpuritySolver(10, {'u': range(2)}, 1025, 10001, 50)

    """
    def test_ImpuritySolver_prepare(self):
        solver = ImpuritySolver(10, {'u': range(2)}, n_iw = 100)
        h = c_dag('u', 0) * c('u', 0)
        g = BlockGf(name_list = ['u'], block_list = [GfImFreq(indices = range(2), beta = 10, n_points = 100)])
        g['u'] << SemiCircular(1)
        solver.prepare(g, h, n_cycles = 50, length_cycle = 2, n_warmup_cycles = 20, verbosity = 0)
    """

    def test_ImpuritySolver_run(self):
        solver = ImpuritySolver(10, {'u': range(2)}, 100, 1000, 25)
        h = c_dag('u', 0) * c('u', 0)
        g = BlockGf(name_list = ['u'],
                    block_list = [GfImFreq(indices = range(2),
                                           beta = 10, n_points = 100)])
        g['u'] << SemiCircular(1)
        solver.run(g, h, n_cycles = 50, length_cycle = 2, n_warmup_cycles = 20, verbosity = 0)
