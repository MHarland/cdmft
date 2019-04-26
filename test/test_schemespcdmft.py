import unittest, numpy as np, os, itertools as itt

from cdmft.parameters import TestDMFTParameters
from cdmft.selfconsistency import Cycle
from cdmft.h5interface import Storage
from cdmft.schemes.pcdmft import GLocal, SelfEnergy, WeissField
from cdmft.greensfunctions import MatsubaraGreensFunction
from cdmft.transformation import MatrixTransformation
from cdmft.operators.hubbard import DimerMomentum


class TestSchemesPCDMFT(unittest.TestCase):

    def test_SchemesPCDMFT_init(self):
        spins = ['up', 'dn']
        sites = range(2)

        g_struct = [[s, sites] for s in spins]
        momenta = ['G', 'X']
        impurity_blocknames = [s+'-'+a for s, a in itt.product(spins, momenta)]
        site_transf_mat = dict([(s, np.sqrt(.5) * np.array([[1,1],[1,-1]])) for s in spins])
        new_struct = [[s+'-'+a, [0]] for s, a in itt.product(spins, momenta)]
        up, dn, a, b = spins[0], spins[1], momenta[0], momenta[1]
        reblock_map = {(up,0,0):(up+'-'+a,0,0),(up,1,1):(up+'-'+b,0,0),(dn,0,0):(dn+'-'+a,0,0),
                       (dn,1,1):(dn+'-'+b,0,0)}
        impurity_transformation = MatrixTransformation(g_struct, site_transf_mat, new_struct,
                                                       reblock_map)


        glat_orb_struct = {s: [0] for s in spins}
        r = [[0.], [-1.], [1.]]
        imp_to_lat_r = [{(s,0,0): (s,0,0) for s in spins}, {(s,0,1): (s,0,0) for s in spins}, {(s,1,0): (s,0,0) for s in spins}]
        weights_r = [1.]*3
        hopping_r = [{s:[[t]] for s in spins} for t in [0, -1, -1]]
        nk = 8
        gcluster_orb_struct = {s: sites for s in spins}
        lat_r_to_cluster = {((0.),s,i,i): (s,i,i) for s in spins for i in range(2)}
        lat_r_to_cluster.update({(((0.),(1.)),s,0,0): (s,0,1) for s in spins})
        lat_r_to_cluster.update({(((1.),(0.)),s,0,0): (s,1,0) for s in spins})
        g = GLocal(glat_orb_struct, gcluster_orb_struct, r, weights_r, hopping_r, nk, imp_to_lat_r, lat_r_to_cluster, impurity_transformation, impurity_blocknames, [1]*4, 10, 1000)

        sigmaimp = MatsubaraGreensFunction([s+'-'+m for s, m in itt.product(spins, momenta)], [1]*4, 10, 1000)
        mu = 1.
        mu = g.set(sigmaimp, 1.)
