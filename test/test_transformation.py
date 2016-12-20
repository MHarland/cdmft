import unittest, numpy as np

from bethe.transformation import GfStructTransformationIndex, MatrixTransformation, InterfaceToBlockstructure


class TestTransformation(unittest.TestCase):

    def test_GfStructTransformationIndex(self):
        momenta = ["G", "X"]
        spins = ["up", "dn"]
        gf_struct_new = [[s+"-"+k, range(1)] for s in spins for k in momenta]
        gf_struct_old = [[s, range(2)] for s in spins]
        site_to_momentum = GfStructTransformationIndex(gf_struct_new, gf_struct_old)
        result = {("up", 0): ("up-G", 0),
                  ("up", 1): ("up-X", 0),
                  ("dn", 0): ("dn-G", 0),
                  ("dn", 1): ("dn-X", 0)}
        self.assertEqual(result, site_to_momentum.index_map)

    def test_MatrixTransformation(self):
        momenta = ["G", "X", "Y", "M"]
        sites = range(4)
        spins = ["up", "dn"]
        gf_struct = [[s, sites] for s in spins]
        transf_mat = .5 * np.array([[1,1,1,1],
                                    [1,-1,1,-1],
                                    [1,1,-1,-1],
                                    [1,-1,-1,1]])
        transf_mat = dict([(s, transf_mat) for s in spins])
        gf_struct_new = [[s+"-"+k, range(1)] for s in spins for k in momenta]
        site_to_momentum = MatrixTransformation(gf_struct, transf_mat, gf_struct_new)
        t = np.array([[0,-2,-2,1],[-2,0,1,-2],[-2,1,0,-2],[1,-2,-2,0]])
        t = dict([(s, t) for s in spins])
        eps = site_to_momentum.transform_matrix(t)
        res = {'dn-G': -3, 'up-G': -3, 'dn-M': 5, 'up-M': 5, 'dn-X': -1, 'dn-Y': -1, 'up-X': -1, 'up-Y': -1}
        for n, b in eps.items():
            self.assertEqual(b[0,0], res[n])
        tnew = site_to_momentum.backtransform_matrix(eps)
        for s, b in t.items():
            self.assertTrue(np.allclose(b, tnew[s]))
        up, dn, a, b, c, d = 'up', 'dn', 'G', 'X', 'Y', 'M'
        reblock_map = {(up,0,0): (up+'-'+a,0,0), (up,1,1): (up+'-'+b,0,0),
                       (up,2,2): (up+'-'+c,0,0), (up,3,3): (up+'-'+d,0,0),
                       (dn,0,0): (dn+'-'+a,0,0), (dn,1,1): (dn+'-'+b,0,0),
                       (dn,2,2): (dn+'-'+c,0,0), (dn,3,3): (dn+'-'+d,0,0)}
        site_to_momentum = MatrixTransformation(gf_struct, transf_mat, gf_struct_new, reblock_map)
        eps = site_to_momentum.transform_matrix(t)
        res = {'dn-G': -3, 'up-G': -3, 'dn-M': 5, 'up-M': 5, 'dn-X': -1, 'dn-Y': -1, 'up-X': -1, 'up-Y': -1}
        for n, b in eps.items():
            self.assertEqual(b[0,0], res[n])
        tnew = site_to_momentum.backtransform_matrix(eps)
        for s, b in t.items():
            self.assertTrue(np.allclose(b, tnew[s]))

    def test_InterfaceToBlockstructure(self):
        spins = ["up", "dn"]
        sites = range(4)
        momenta = ["G", "X", "Y", "M"]
        gf_struct = [[s, sites] for s in spins]
        gf_struct_new = [[s+"-"+k, range(1)] for s in spins for k in momenta]
        t = np.array([[0,1,1,2],[1,1,2,1],[1,2,2,1],[2,1,1,3]])
        t = dict([(s, t) for s in spins])
        intf = InterfaceToBlockstructure(t, gf_struct, gf_struct_new)
        for b, i in zip(gf_struct_new, range(4)+range(4)):
            self.assertEqual(intf[b[0], 0, 0], i)
