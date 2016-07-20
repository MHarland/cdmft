import unittest, numpy as np
from pytriqs.gf.local import BlockGf, GfImFreq, SemiCircular

from bethe.weissfield import WeissField, WeissFieldNambu


class TestWeissField(unittest.TestCase):

    def test_WeissField_initialization(self):
        mu = t_loc = dict([[spin, np.identity(2)] for spin in ['u', 'd']])
        t = 1
        g = WeissField(['u', 'd'], [range(2)] * 2, 5, 50, t, t_loc)
        self.assertEqual(g.t_loc['u'][0, 0], 1)

    def test_WeissField_selfconsistency(self):
        mu = t_loc = dict([[spin, np.identity(2)] for spin in ['u', 'd']])
        t = 1
        g = WeissField(['u', 'd'], [range(2)] * 2, 5, 50, t, t_loc)
        g2 = g.gf.copy()
        g.calc_selfconsistency(g2, mu_number = 2)

    def test_WeissField_set_mu(self):
        mu = t_loc = dict([[spin, np.identity(2)] for spin in ['u', 'd']])
        t = 1
        g = WeissField(['u', 'd'], [range(2)] * 2, 5, 50, t, t_loc)
        g.set_mu(0.3)
        self.assertEqual(g.mu['u'][1,1], 0.3)

    def test_WeissFieldNambu(self):
        mu = dict([[spin, np.array([[-1,0],[0,-1]])] for spin in ['g', 'x']])
        t_loc = dict([[spin, np.identity(2)] for spin in ['g', 'x']])
        t = 1
        gw = WeissFieldNambu(['g', 'x'], [range(2)] * 2, 5, 50, t, t_loc)
        g = BlockGf(name_list = ['g', 'x'],
                    block_list = [GfImFreq(indices = range(2),
                                           beta = 5, n_points = 50)] * 2)
        for block in ['x', 'g']:
            g[block] << SemiCircular(1)
        g['x'][0, 1] << 0.01
        g['x'][1, 0] << 0.01
        gw.calc_selfconsistency(g, mu)
