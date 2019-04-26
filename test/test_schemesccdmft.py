import unittest, numpy as np, os, itertools as itt

from cdmft.parameters import TestDMFTParameters
from cdmft.selfconsistency import Cycle
from cdmft.h5interface import Storage
from cdmft.schemes.ccdmft import GLocal, SelfEnergy, WeissField, HoppingLattice
from cdmft.greensfunctions import MatsubaraGreensFunction
from cdmft.transformation import MatrixTransformation
#from cdmft.tightbinding import LatticeDispersion
from cdmft.operators.hubbard import DimerMomentum


class TestSchemesCCDMFT(unittest.TestCase):

    def test_SchemesCCDMFT_init(self):
        spins = ['up', 'dn']
        sites = range(2)
        momenta = ['G', 'X']
        
        #lattice        
        g_struct = [[s, sites] for s in spins]
        glat_orb_struct = {s: [0] for s in spins}
        r = [[0.], [-1.], [1.]]
        imp_to_lat_r = [{(s,0,0): (s,0,0) for s in spins}, {(s,0,1): (s,0,0) for s in spins}, {(s,1,0): (s,0,0) for s in spins}]
        weights_r = [1.]*3
        hopping_r = [{s:[[t]] for s in spins} for t in [0, -1, -1]]
        nk = 8

        #cluster
        gcluster_orb_struct = {s: sites for s in spins}
        lat_r_to_cluster = {}
        lat_r_to_cluster.update({((0.),(0.),s,0,0): (s,0,0) for s in spins})
        lat_r_to_cluster.update({((0.),(1.),s,0,0): (s,0,1) for s in spins})
        lat_r_to_cluster.update({((1.),(0.),s,0,0): (s,1,0) for s in spins})
        lat_r_to_cluster.update({((1.),(1.),s,0,0): (s,1,1) for s in spins})

        #cluster and environment
        r_cavity = [[-1],[2]]
        r_cluster = [[0],[1]]

        #impurity
        impurity_blocknames = [s+'-'+a for s, a in itt.product(spins, momenta)]
        site_transf_mat = dict([(s, np.sqrt(.5) * np.array([[1,1],[1,-1]])) for s in spins])
        new_struct = [[s+'-'+a, [0]] for s, a in itt.product(spins, momenta)]
        up, dn, a, b = spins[0], spins[1], momenta[0], momenta[1]
        reblock_map = {(up,0,0):(up+'-'+a,0,0),(up,1,1):(up+'-'+b,0,0),(dn,0,0):(dn+'-'+a,0,0),
                       (dn,1,1):(dn+'-'+b,0,0)}
        impurity_transformation = MatrixTransformation(g_struct, site_transf_mat, new_struct,
                                                       reblock_map)
        
        gloc = GLocal(glat_orb_struct, gcluster_orb_struct, r, weights_r, hopping_r, nk, imp_to_lat_r, lat_r_to_cluster, impurity_transformation, r_cavity, r_cluster, impurity_blocknames, [1]*4, 10, 1000)

        #sigmaimp = BlockGf(name_block_generator = [(s, GfImFreq(beta = 10, n_points = 1001, indices = [0])) for s in spins])
        #sigmaimp = MatsubaraGreensFunction(spins, [2, 2], 10, 1000)
        sigmaimp = MatsubaraGreensFunction([s+'-'+m for s, m in itt.product(spins, momenta)], [1]*4, 10, 1000)
        mu = 1.
        mu = gloc.set(sigmaimp, 1.)

    def test_HoppingLattice(self):
        spins = ['up', 'dn']
        r = [[0.], [-1.], [1.]]
        hopping_r = [{s:[[t]] for s in spins} for t in [2, -1, -1]]
        h = HoppingLattice(r, hopping_r)
        self.assertEqual(h[[3],[3]]['up'][0,0], 2)
        self.assertEqual(h[[3],[4]]['up'][0,0], -1)
        self.assertEqual(h[[4],[3]]['up'][0,0], -1)
        self.assertEqual(h[[5],[3]]['up'][0,0], 0)

