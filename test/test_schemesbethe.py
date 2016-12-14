import unittest, numpy as np

from bethe.schemes.bethe import GLocal, SelfEnergy, WeissField


class TestSchemesBethe(unittest.TestCase):

    def test_SchemesBethe_init(self):
        h = np.array([[0,0],[0,0]])
        g = GLocal(1, {'up': h, 'dn': h}, ['up', 'dn'], [2, 2], 10, 1001)

    def test_SchemesBethe_calculate(self):
        h = np.array([[0]])
        g = GLocal(1, {'up': h, 'dn': h}, ['up', 'dn'], [1, 1], 10, 1001)
        se = SelfEnergy(['up', 'dn'], [1, 1], 10, 1001)
        se.zero()
        g.set(se, 0, 200, 1001, 8)
        g.calculate(se,0, 200, 1001, 8)

    def test_SchemesBethe_find_and_set_mu_single(self):
        h = np.array([[0]])
        g = GLocal(1, {'up': h, 'dn': h}, ['up', 'dn'], [1, 1], 10, 1001)
        se = SelfEnergy(['up', 'dn'], [1, 1], 10, 1001)
        se.zero()
        g.set(se, 0, 200, 1001, 8)
        self.assertTrue(abs(g.total_density()-1) < 1e-3)
        g.find_and_set_mu(1, se, 0.23, 100, 200, 1001, 8)
        self.assertTrue(abs(g.total_density()-1) < 1e-3)

    def test_SchemesBethe_find_and_set_mu_double(self):
        h = np.array([[-1,0],[0,1]])
        g = GLocal(1, {'up': h, 'dn': h}, ['up', 'dn'], [2, 2], 10, 1001)
        se = SelfEnergy(['up', 'dn'], [2, 2], 10, 1001)
        se.zero()
        g.find_and_set_mu(2, se, .12, 100, 800, 1001, 3)
        self.assertTrue(abs(g.total_density()-2) < 1e-2)
