import numpy as np, itertools as itt

from bethe.setups.generic import CycleSetupGeneric
from bethe.operators.hubbard import DimerMomentum
from bethe.operators.kanamori import Dimer as KanamoriDimer
from bethe.schemes.cdmft import GLocal, SelfEnergy, WeissField
from bethe.tightbinding import LatticeDispersion, LatticeDispersionMultiband
from bethe.transformation import MatrixTransformation


class MomentumDimerSetup(CycleSetupGeneric):
    """
    Dimer-cluster of a 1D chain lattice within Cluster DMFT
    assumes that transformation_matrix diagonalizes the GLocal on site-space
    i.e. the two clustersites are equivalent
    """
    def __init__(self, beta, mu, u, t, n_k, spins = ['up', 'dn'], momenta = ['+', '-'], n_iw = 1025):
        clusterhopping = {(0): [[0,t],[t,0]], (1): [[0,t],[0,0]], (-1): [[0,0],[t,0]]}
        disp = LatticeDispersion(clusterhopping, n_k)
        site_transf_mat = dict([(s, np.sqrt(.5) * np.array([[1,1],[1,-1]])) for s in spins])
        new_struct = [[s+'-'+a, [0]] for s, a in itt.product(spins, momenta)]
        up, dn, a, b = spins[0], spins[1], momenta[0], momenta[1]
        reblock_map = {(up,0,0):(up+'-'+a,0,0),(up,1,1):(up+'-'+b,0,0),(dn,0,0):(dn+'-'+a,0,0),
                       (dn,1,1):(dn+'-'+b,0,0)}
        disp.transform_site_space(site_transf_mat, new_struct, reblock_map)
        hubbard = DimerMomentum(u, spins, momenta, site_transf_mat)
        self.h_int = hubbard.get_h_int()
        self.gloc = GLocal(disp, [s[0] for s in new_struct], [1] * 4, beta, n_iw)
        self.g0 = WeissField([s[0] for s in new_struct], [1] * 4, beta, n_iw)
        self.se = SelfEnergy([s[0] for s in new_struct], [1] * 4, beta, n_iw)
        self.mu = mu
        self.global_moves = {"spin-flip": dict([((s1+"-"+k, 0), (s2+"-"+k, 0)) for k in momenta for s1, s2 in itt.product(spins, spins) if s1 != s2])}
        self.quantum_numbers = [hubbard.get_n_tot(), hubbard.get_n_per_spin(up)]
        
        
class StrelSetup(CycleSetupGeneric):
    """
    TODO
    """
    def __init__(self, beta, mu, t_d, t_c, tp, u, j, n_k, spins = ['up', 'dn'], orbs = ['d', 'c'], sites = range(2), n_iw = 1025):
        struct = [(s+'-'+orb, sites) for s, orb in itt.product(spins, orbs)]
        clusterhopping_d = {(0): [[0,t_d],[t_d,0]], (1): [[0,tp],[0,0]], (-1): [[0,0],[tp,0]]}
        clusterhopping_c = {(0): [[0,t_c],[t_c,0]], (1): [[0,tp],[0,0]], (-1): [[0,0],[tp,0]]}
        disp_d = LatticeDispersion(clusterhopping_d, n_k)
        disp_c = LatticeDispersion(clusterhopping_c, n_k)
        disp = LatticeDispersionMultiband({'up-d': disp_d, 'dn-d': disp_d,
                                           'up-c': disp_c, 'dn-c': disp_c})
        hamiltonian = KanamoriDimer(u, j)
        self.h_int = hamiltonian
        self.gloc = GLocal(disp, gf_struct = struct, beta = beta, n_iw = n_iw)
        self.g0 = WeissField(gf_struct = struct, beta = beta, n_iw = n_iw)
        self.se = SelfEnergy(gf_struct = struct, beta = beta, n_iw = n_iw)
        self.mu = mu
        self.global_moves = {"spin-flip": {((s1+'-'+orb, i), (s2+'-'+orb, i)) for i, orb in zip(sites, orbs) for s1, s2 in itt.product(spins, spins) if s1 != s2},
                             "dimer-flip": {((s+'-'+orb, i1), (s+'-'+orb, i2)) for s, orb in zip(spins, orbs) for i1, i2 in itt.product(sites, sites) if i1 != i2}}
        self.quantum_numbers = []
