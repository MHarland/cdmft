import numpy as np
import itertools as itt

from cdmft.operators.hubbard import DimerMomentum
from cdmft.schemes.pcdmft import GLocal, SelfEnergy, WeissField
from cdmft.setups.generic import CycleSetupGeneric
from cdmft.transformation import MatrixTransformation


class DimerChainSetup(CycleSetupGeneric):
    """
    TODO needs testing
    """
    def __init__(self, beta, mu, u, n_k, spins = ['up', 'dn'], n_iw = 1025):
        sites = range(2)
        hubbard = DimerMomentum(u, spins)
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
        gcluster_orb_struct = {s: sites for s in spins}
        lat_r_to_cluster = {(((0.),(0.)),s,0,0): (s,i,i) for s in spins for i in range(2)}
        lat_r_to_cluster.update({(((0.),(1.)),s,0,0): (s,0,1) for s in spins})
        lat_r_to_cluster.update({(((1.),(0.)),s,0,0): (s,1,0) for s in spins})
        
        self.h_int = hubbard.get_h_int()
        self.gloc = GLocal(glat_orb_struct, gcluster_orb_struct, r, weights_r, hopping_r, n_k, imp_to_lat_r,
                           lat_r_to_cluster, impurity_transformation, impurity_blocknames, [1]*4, beta, n_iw)
        self.se = SelfEnergy([s+'-'+m for s, m in itt.product(spins, momenta)], [1]*4, beta, n_iw)
        self.g0 = WeissField([s+'-'+m for s, m in itt.product(spins, momenta)], [1]*4, beta, n_iw)
        self.mu = mu
        self.global_moves = {}
        self.quantum_numbers = [hubbard.n_tot(), hubbard.sz_tot()]
