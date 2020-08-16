import unittest
import numpy as np
from pytriqs.gf import BlockGf, GfImFreq

from cdmft.transformation2 import Transformation, Reblock, UnitaryMatrixTransformation


class TestTransformation2(unittest.TestCase):

    def test_ReblockG(self):
        momenta = ["G", "X"]
        spins = ["up", "dn"]
        struct_new = [[s+"-"+k, range(1)] for s in spins for k in momenta]
        struct_old = [[s, range(2)] for s in spins]
        rbmap = {('up', 0, 0): ('up-G', 0, 0), ('dn', 0, 1): ('up-X', 0, 0)}
        g = BlockGf(name_list=[b[0] for b in struct_old], block_list=[
                    GfImFreq(indices=b[1], n_points=100, beta=10) for b in struct_old])
        g['up'][0, 0] << 1.
        g['dn'][0, 1] << 2.
        reblg = Reblock(struct_new, struct_old, rbmap)
        g = reblg(g)
        self.assertEqual(g['up-G'].data[0, 0, 0], 1.)
        self.assertEqual(g['up-X'].data[0, 0, 0], 2.)
        g = reblg.inverse(g)
        self.assertEqual(g['up'].data[0, 0, 0], 1.)
        self.assertEqual(g['dn'].data[0, 0, 1], 2.)

    def test_Transformation(self):
        momenta = ["G", "X", "Y", "M"]
        sites = range(4)
        spins = ["up", "dn"]
        struct_old = [[s, sites] for s in spins]
        transf_mat = .5 * np.array([[1, 1, 1, 1],
                                    [1, -1, 1, -1],
                                    [1, 1, -1, -1],
                                    [1, -1, -1, 1]])
        transf_mat = dict([(s, transf_mat) for s in spins])
        struct_new = [[s+"-"+k, range(1)] for s in spins for k in momenta]

        g = BlockGf(name_list=[b[0] for b in struct_old], block_list=[
                    GfImFreq(indices=b[1], n_points=100, beta=10) for b in struct_old])
        site_to_mom = UnitaryMatrixTransformation(transf_mat)
        transf = Transformation([site_to_mom])
        g = transf.transform(g)
        g = transf.backtransform(g)

        g['up'][0, 1] << -2.
        g['up'][0, 2] << -2.
        g['up'][0, 3] << 1.
        g['up'][1, 0] << -2.
        g['up'][1, 2] << 1.
        g['up'][1, 3] << -2.
        g['up'][2, 0] << -2.
        g['up'][2, 1] << 1.
        g['up'][2, 3] << -2.
        g['up'][3, 0] << 1.
        g['up'][3, 1] << -2.
        g['up'][3, 2] << -2.
        up, dn, a, b, c, d = 'up', 'dn', 'G', 'X', 'Y', 'M'
        reblock_map = {(up, 0, 0): (up+'-'+a, 0, 0), (up, 1, 1): (up+'-'+b, 0, 0),
                       (up, 2, 2): (up+'-'+c, 0, 0), (up, 3, 3): (up+'-'+d, 0, 0),
                       (dn, 0, 0): (dn+'-'+a, 0, 0), (dn, 1, 1): (dn+'-'+b, 0, 0),
                       (dn, 2, 2): (dn+'-'+c, 0, 0), (dn, 3, 3): (dn+'-'+d, 0, 0)}
        reblock = Reblock(struct_new, struct_old, reblock_map)
        transf = Transformation([site_to_mom, reblock])
        g = transf.transform(g)
        self.assertEqual(g['up-G'].data[0, 0, 0], -3.)
        self.assertEqual(g['up-M'].data[0, 0, 0], 5.)
        self.assertEqual(g['up-X'].data[0, 0, 0], -1.)
        self.assertEqual(g['up-Y'].data[0, 0, 0], -1.)
        g = transf.backtransform(g)
        self.assertEqual(g['up'].data[0, 0, 1], -2.)
