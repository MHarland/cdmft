import unittest, numpy as np, itertools as itt

from bethe.tightbinding import LatticeDispersion


class TestTightbinding(unittest.TestCase):

    def test_LatticeDispersion_dimer_in_chain(self):
        t = -1
        h = {(0, 0): [[0,t],[t,0]],(1, 0): [[0,t],[0,0]],(-1, 0): [[0,0],[t,0]]}
        disp = LatticeDispersion(h, 10)
        t = np.sum([w * d['up'] for k, w, d in disp.loop_over_bz()], axis = 0)
        self.assertTrue(np.allclose(t, h[(0, 0)]))

    def test_LatticeDispersion_dimer_in_chain_transform(self):
        t = -1
        h = {(0, 0): [[0,t],[t,0]],(1, 0): [[0,t],[0,0]],(-1, 0): [[0,0],[t,0]]}
        disp = LatticeDispersion(h, 10)
        u = np.sqrt(.5) * np.array([[1,1],[1,-1]])
        u = {'up': u, 'dn': u}
        new_struct = [[s+'-'+a, [0]] for s, a in itt.product(['up', 'dn'], ['+', '-'])]
        reblock_map = {('up',0,0):('up-+',0,0),('up',1,1):('up--',0,0),('dn',0,0):('dn-+',0,0)
                       ,('dn',1,1):('dn--',0,0)}
        disp.transform_site_space(u, new_struct, reblock_map)
        t = dict()
        for orb, result in zip(new_struct, [-1, 1, -1, 1]):
            orbname = orb[0]
            t[orbname] = np.sum([w * d[orbname] for k, w, d in disp.loop_over_bz()], axis = 0)
            self.assertTrue(np.allclose(t[orbname], result))
