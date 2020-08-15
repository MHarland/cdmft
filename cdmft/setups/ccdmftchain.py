import numpy as np, itertools as itt

from cdmft.setups.common import CycleSetupCommon
from cdmft.operators.hubbard import DimerMomentum
from cdmft.schemes.ccdmft import GLocal, SelfEnergy, WeissField
from cdmft.transformation import MatrixTransformation


class MomentumDimerSetup(CycleSetupCommon):
    """
    Dimer-cluster of a 1D chain lattice within Cluster DMFT
    assumes that transformation_matrix diagonalizes the GLocal on site-space
    i.e. the two clustersites are equivalent
    """
    def __init__(self, beta, mu, u, t, n_k, spins = ['up', 'dn'], momenta = ['G', 'X'], sites = range(2), n_iw = 1025):
        
        #lattice        
        g_struct = [[s, sites] for s in spins]
        glat_orb_struct = {s: [0] for s in spins}
        r = [[0.], [-1.], [1.]]
        imp_to_lat_r = [{(s,0,0): (s,0,0) for s in spins}, {(s,0,1): (s,0,0) for s in spins}, {(s,1,0): (s,0,0) for s in spins}]
        weights_r = [1.]*3
        hopping_r = [{s:[[t]] for s in spins} for t in [0, -1, -1]]

        #cluster
        gcluster_orb_struct = {s: sites for s in spins}
        lat_r_to_cluster = {}
        lat_r_to_cluster.update({((0.),(0.),s,0,0): (s,0,0) for s in spins})
        lat_r_to_cluster.update({((0.),(1.),s,0,0): (s,0,1) for s in spins})
        lat_r_to_cluster.update({((1.),(0.),s,0,0): (s,1,0) for s in spins})
        lat_r_to_cluster.update({((1.),(1.),s,0,0): (s,1,1) for s in spins})
        one = np.eye(1,1)
        tcluster = {'up-G': -one, 'dn-G': -one, 'up-X': +one, 'dn-X': +one}

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
        hubbard = DimerMomentum(u, spins, momenta, site_transf_mat)
        
        self.h_int = hubbard.get_h_int()
        self.gloc = GLocal(glat_orb_struct, gcluster_orb_struct, r, weights_r, hopping_r, n_k, imp_to_lat_r, lat_r_to_cluster, impurity_transformation, r_cavity, r_cluster, impurity_blocknames, [1]*4, 10, n_iw)
        self.se = SelfEnergy([s+'-'+m for s, m in itt.product(spins, momenta)], [1] * 4, beta, n_iw)
        self.g0 = WeissField(tcluster, [s[0] for s in new_struct], [1] * 4, beta, n_iw)
        self.mu = mu
        self.global_moves = {}#{"spin-flip": dict([((s1+"-"+k, 0), (s2+"-"+k, 0)) for k in momenta for s1, s2 in itt.product(spins, spins) if s1 != s2])}
        self.quantum_numbers = [hubbard.n_tot(), hubbard.sz_tot()]
