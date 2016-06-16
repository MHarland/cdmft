import unittest, numpy as np

from Bethe.weissfield import WeissField


class TestWeissField(unittest.TestCase):

    def test_WeissField_initialization(self):
        mu = t_loc = t = dict([[spin, np.identity(2)] for spin in ['u', 'd']])
        g = WeissField(['u', 'd'], [range(2)] * 2, 5, 50, t, mu, t_loc)
        g = WeissField(['u', 'd'], [range(2)] * 2, 5, 50, t)
        self.assertEqual(g.mu['u'][0, 0], 0)

    def test_WeissField_selfconsistency(self):
        mu = t_loc = t = dict([[spin, np.identity(2)] for spin in ['u', 'd']])
        g = WeissField(['u', 'd'], [range(2)] * 2, 5, 50, t, mu, t_loc)
        g2 = g.gf.copy()
        g.calc_selfconsistency(g2)

    def test_WeissField_set_mu(self):
        mu = t_loc = t = dict([[spin, np.identity(2)] for spin in ['u', 'd']])
        g = WeissField(['u', 'd'], [range(2)] * 2, 5, 50, t, mu, t_loc)
        g.set_mu(0.3)
        self.assertEqual(g.mu['u'][1,1], 0.3)
