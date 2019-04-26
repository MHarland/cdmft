import unittest, numpy as np, itertools as itt

from cdmft.schemes.bethe import GLocal, SelfEnergy, WeissField, GLocalAFM, WeissFieldAIAO, GLocalWithOffdiagonals, GLocalAIAO


class TestSchemesBethe(unittest.TestCase):

    def test_SchemesBethe_init(self):
        h = np.array([[0,0],[0,0]])
        g = GLocal(1, {'up': h, 'dn': h}, None, None, 3, ['up', 'dn'], [2, 2], 10, 1001)

    def test_SchemesBethe_calculate(self):
        h = np.array([[0]])
        g = GLocal(1, {'up': h, 'dn': h}, None, None, 3, ['up', 'dn'], [1, 1], 10, 1001)
        se = SelfEnergy(['up', 'dn'], [1, 1], 10, 1001)
        se.zero()
        g.set(se, 0)

    def test_SchemesBetheAFM_calculate(self):
        h = np.array([[0]])
        g = GLocalAFM(1, {'up': h, 'dn': h}, None, None, 3, ['up', 'dn'], [1, 1], 10, 1001)
        se = SelfEnergy(['up', 'dn'], [1, 1], 10, 1001)
        se.zero()
        g.set(se, 0)

    def test_SchemesBetheAIAO(self):
        g = GLocalAIAO(1, {'spin-site': np.identity(6)}, ['spin-site'], [6], 10, 1001)
        se = SelfEnergy(['spin-site'], [6], 10, 1001)
        se.zero()
        testmat = np.array([[i*j for j in range(1,7)] for i in range(1,7)])
        g['spin-site'].data[1001,:,:] = testmat
        g0 = WeissFieldAIAO(['spin-site'], [6], 10, 1001)
        g0.calc_selfconsistency(g, se, 3)
        g.find_and_set_mu(3., se, 0, 1000)
        self.assertTrue(g._last_g_loc_convergence[-1] < 0.001)

    def test_SchemesBethe_find_and_set_mu_single(self):
        h = np.array([[0]])
        g = GLocal(1, {'up': h, 'dn': h}, None, None, 3, ['up', 'dn'], [1, 1], 10, 1001)
        se = SelfEnergy(['up', 'dn'], [1, 1], 10, 1001)
        se.zero()
        g.set(se, 0)
        self.assertTrue(abs(g.total_density()-1) < 1e-3)
        g.find_and_set_mu(1, se, 0.23, 100)
        self.assertTrue(abs(g.total_density()-1) < 1e-3)

    def test_SchemesBethe_find_and_set_mu_double(self):
        h = np.array([[-1,0],[0,1]])
        g = GLocal(1, {'up': h, 'dn': h}, None, None, 3, ['up', 'dn'], [2, 2], 10, 1001)
        se = SelfEnergy(['up', 'dn'], [2, 2], 10, 1001)
        se.zero()
        g.find_and_set_mu(2, se, .12, 3)
        self.assertTrue(abs(g.total_density()-2) < 1e-2)
