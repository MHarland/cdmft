import unittest, numpy as np, os, itertools as itt

from cdmft.parameters import TestDMFTParameters
from cdmft.selfconsistency import Cycle
from cdmft.h5interface import Storage
from cdmft.schemes.cdmft import GLocal, SelfEnergy, WeissField
from cdmft.tightbinding import LatticeDispersion
from cdmft.operators.hubbard import DimerMomentum


class TestSchemesCDMFT(unittest.TestCase):

    def test_SchemesCDMFT_init(self):
        t = -1
        h = {(0, 0): [[0,t],[t,0]],(1, 0): [[0,t],[0,0]],(-1, 0): [[0,0],[t,0]]}
        disp = LatticeDispersion(h, 8)
        g = GLocal(disp, None, ['up', 'dn'], [2, 2], 10, 1000)

    def test_SchemesCDMFT_dmu(self):
        t = -1
        h = {(0, 0): [[0,t],[t,0]],(1, 0): [[0,t],[0,0]],(-1, 0): [[0,0],[t,0]]}
        disp = LatticeDispersion(h, 8)
        g = GLocal(disp, None, ['up', 'dn'], [2, 2], 10, 1000)
        se = SelfEnergy(['up', 'dn'], [2, 2], 10, 1000)
        mu = g.set(se, 3)
        self.assertEqual(3, mu)

    def test_SchemesCDMFT_calculate_clustersite_basis(self):
        t = -1
        h = {(0, 0): [[0,t],[t,0]],(1, 0): [[0,t],[0,0]],(-1, 0): [[0,0],[t,0]]}
        disp = LatticeDispersion(h, 8)
        g = GLocal(disp, None, ['up', 'dn'], [2, 2], 10, 1000)
        se = SelfEnergy(['up', 'dn'], [2, 2], 10, 1000)
        se.zero()
        g.set(se, 0)
        self.assertAlmostEqual(g.total_density(), 2)

    def test_SchemesCDMFT_calculate_clustermomentum_basis(self):
        t = -1
        h = {(0, 0): [[0,t],[t,0]],(1, 0): [[0,t],[0,0]],(-1, 0): [[0,0],[t,0]]}
        disp = LatticeDispersion(h, 8)
        u = np.sqrt(.5) * np.array([[1,1],[1,-1]])
        u = {'up': u, 'dn': u}
        new_struct = [[s+'-'+a, [0]] for s, a in itt.product(['up', 'dn'], ['+', '-'])]
        reblock_map = {('up',0,0):('up-+',0,0),('up',1,1):('up--',0,0),('dn',0,0):('dn-+',0,0)
                       ,('dn',1,1):('dn--',0,0)}
        disp.transform_site_space(u, new_struct, reblock_map)
        g = GLocal(disp, None, ['up-+', 'up--', 'dn-+', 'dn--'], [1] * 4, 10, 1000)
        se = SelfEnergy(['up-+', 'up--', 'dn-+', 'dn--'], [1] * 4, 10, 1000)
        g.set(se, 0)
        self.assertAlmostEqual(g.total_density(), 2)

    def test_SchemesCDMFT_Cycle(self):
        t = -1
        h = {(0, 0): [[0,t],[t,0]],(1, 0): [[0,t],[0,0]],(-1, 0): [[0,0],[t,0]]}
        disp = LatticeDispersion(h, 8)
        u = np.sqrt(.5) * np.array([[1,1],[1,-1]])
        u = {'up': u, 'dn': u}
        new_struct = [[s+'-'+a, [0]] for s, a in itt.product(['up', 'dn'], ['+', '-'])]
        reblock_map = {('up',0,0):('up-+',0,0),('up',1,1):('up--',0,0),('dn',0,0):('dn-+',0,0),
                       ('dn',1,1):('dn--',0,0)}
        disp.transform_site_space(u, new_struct, reblock_map)
        gloc = GLocal(disp, None, ['up-+', 'up--', 'dn-+', 'dn--'], [1] * 4, 10, 1000)
        se = SelfEnergy(['up-+', 'up--', 'dn-+', 'dn--'], [1] * 4, 10, 1000)
        g0 = WeissField(['up-+', 'up--', 'dn-+', 'dn--'], [1] * 4, 10, 1000)
        sto = Storage("test.h5")
        params = TestDMFTParameters(filling = 2)
        h = DimerMomentum(4)
        cyc = Cycle(sto, params, h.get_h_int(), gloc, g0, se, 2)
        cyc.run(1)
        os.remove("test.h5")
