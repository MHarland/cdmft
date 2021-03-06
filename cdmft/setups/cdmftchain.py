import numpy as np, itertools as itt

from cdmft.setups.common import CycleSetupCommon
from cdmft.operators.hubbard import DimerMomentum, Site, Dimer
from cdmft.operators.kanamori import Dimer as KanamoriDimer
from cdmft.schemes.cdmft import GLocal, SelfEnergy, WeissField
from cdmft.tightbinding import LatticeDispersion, LatticeDispersionMultiband
from cdmft.transformation import MatrixTransformation


class SingleSiteSetup(CycleSetupCommon):
    """
    Dimer-cluster of a 1D chain lattice within Cluster DMFT
    assumes that transformation_matrix diagonalizes the GLocal on site-space
    i.e. the two clustersites are equivalent
    """
    def __init__(self, beta, mu, u, t, n_k, spins = ['up', 'dn'], n_iw = 1025):
        clusterhopping = {(1): [[t]], (-1): [[t]]}
        disp = LatticeDispersion(clusterhopping, n_k)
        site_transf_mat = dict([(s, np.sqrt(.5) * np.array([[1,1],[1,-1]])) for s in spins])
        struct = [[s, [0]] for s in spins]
        up, dn = spins[0], spins[1]
        hubbard = Site(u, spins)
        self.h_int = hubbard.get_h_int()
        self.gloc = GLocal(disp, None, [s[0] for s in struct], [1] * 2, beta, n_iw)
        self.g0 = WeissField([s[0] for s in struct], [1] * 2, beta, n_iw)
        self.se = SelfEnergy([s[0] for s in struct], [1] * 2, beta, n_iw)
        self.mu = mu
        self.global_moves = {}#{"spin-flip": dict([((s1, 0), (s2, 0)) for s1, s2 in itt.product(spins, spins) if s1 != s2])}
        self.quantum_numbers = [hubbard.n_tot(), hubbard.sz_tot()]


class MomentumDimerSetup(CycleSetupCommon):
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
        self.gloc = GLocal(disp, None, [s[0] for s in new_struct], [1] * 4, beta, n_iw)
        self.g0 = WeissField([s[0] for s in new_struct], [1] * 4, beta, n_iw)
        self.se = SelfEnergy([s[0] for s in new_struct], [1] * 4, beta, n_iw)
        self.mu = mu
        self.global_moves = {}#{"spin-flip": dict([((s1+"-"+k, 0), (s2+"-"+k, 0)) for k in momenta for s1, s2 in itt.product(spins, spins) if s1 != s2])}
        self.quantum_numbers = [hubbard.n_tot(), hubbard.sz_tot()]


class DimerSetup(CycleSetupCommon):
    """
    Dimer-cluster of a 1D chain lattice within Cluster DMFT
    assumes that transformation_matrix diagonalizes the GLocal on site-space
    i.e. the two clustersites are equivalent
    """
    def __init__(self, beta, mu, u, t, n_k, spins = ['up', 'dn'], n_iw = 1025, site_transf_mat = dict([(s, np.sqrt(.5) * np.array([[1,1],[1,-1]])) for s in ['up', 'dn']])):
        clusterhopping = {(0): [[0,t],[t,0]], (1): [[0,t],[0,0]], (-1): [[0,0],[t,0]]}
        disp = LatticeDispersion(clusterhopping, n_k)
        if site_transf_mat is not None:
            disp.transform_site_space(site_transf_mat)
        hubbard = Dimer(u, spins, site_transf_mat)
        self.h_int = hubbard.get_h_int()
        self.gloc = GLocal(disp, None, [s for s in spins], [2] * 2, beta, n_iw)
        self.g0 = WeissField([s for s in spins], [2] * 2, beta, n_iw)
        self.se = SelfEnergy([s for s in spins], [2] * 2, beta, n_iw)
        self.mu = mu
        self.global_moves = {}#{"spin-flip": dict([((s1+"-"+k, 0), (s2+"-"+k, 0)) for k in momenta for s1, s2 in itt.product(spins, spins) if s1 != s2])}
        self.quantum_numbers = [hubbard.n_tot(), hubbard.sz_tot()]

        
class StrelSetup(CycleSetupCommon):
    """
    TODO
    """
    def __init__(self, beta, mu, t_d, t_c, tp, u, j, n_k, site_transf_d = 0, site_transf_c = 0, spins = ['up', 'dn'], orbs = ['d', 'c'], sites = range(2), n_iw = 1025, density_density_only = False):
        self.orbs = orbs
        self.spins = spins
        self.site_transf_d = site_transf_d
        self.site_transf_c = site_transf_c
        self.nnonly = density_density_only
        struct = [(s+'-'+orb, sites) for s, orb in itt.product(spins, orbs)]
        clusterhopping_d = {(0): [[0,t_d],[t_d,0]], (1): [[0,tp],[0,0]], (-1): [[0,0],[tp,0]]}
        clusterhopping_c = {(0): [[0,t_c],[t_c,0]], (1): [[0,tp],[0,0]], (-1): [[0,0],[tp,0]]}
        disp_d = LatticeDispersion(clusterhopping_d, n_k)
        disp_c = LatticeDispersion(clusterhopping_c, n_k)
        if site_transf_d != 0:
            disp_d.transform_site_space(self._site_transf_mat(site_transf_d))
        if site_transf_c != 0:
            disp_c.transform_site_space(self._site_transf_mat(site_transf_c))
        disp = LatticeDispersionMultiband({spins[0]+'-'+orbs[0]: disp_d,
                                           spins[1]+'-'+orbs[0]: disp_d,
                                           spins[0]+'-'+orbs[1]: disp_c,
                                           spins[1]+'-'+orbs[1]: disp_c})
        self.h_int = KanamoriDimer(u, j, transf = self._site_transf(site_transf_d, site_transf_c), density_density_only = self.nnonly)
        self.gloc = GLocal(disp, None, gf_struct = struct, beta = beta, n_iw = n_iw)
        self.g0 = WeissField(gf_struct = struct, beta = beta, n_iw = n_iw)
        self.se = SelfEnergy(gf_struct = struct, beta = beta, n_iw = n_iw)
        self.mu = mu
        self.global_moves = {}#{"spin-flip": {((s1+'-'+orb, i), (s2+'-'+orb, i)) for i, orb in zip(sites, orbs) for s1, s2 in itt.product(spins, spins) if s1 != s2}, "dimer-flip": {((s+'-'+orb, i1), (s+'-'+orb, i2)) for s, orb in zip(spins, orbs) for i1, i2 in itt.product(sites, sites) if i1 != i2}} #here, dimer-flip would not work with a site-transformation, could be implemented if needed, remeber to reset it if transform_sites
        self.quantum_numbers = [self.h_int.n_tot()]#, self.h_int.s2_tot()]

    def transform_sites(self, angle_d, angle_c):
        change_d = angle_d - self.site_transf_d
        change_c = angle_c - self.site_transf_c
        self.site_transf_d = angle_d
        self.site_transf_c = angle_c
        self.gloc.lat.transform_site_space(self._site_transf(change_d, change_c),
                                           [s+'-'+self.orbs[1] for s in self.spins])
        self.gloc.lat.transform_site_space(self._site_transf(change_d, change_c),
                                           [s+'-'+self.orbs[0] for s in self.spins])
        self.h_int = KanamoriDimer(self.h_int.u, self.h_int.j, transf = self._site_transf(angle_d, angle_c), density_density_only = self.nnonly)
        transf_d = MatrixTransformation(self.gloc.gf_struct, self._site_transf(angle_d, angle_c),
                                        orbital_filter = [s+'-'+self.orbs[1] for s in self.spins])
        transf_c = MatrixTransformation(self.gloc.gf_struct, self._site_transf(angle_d, angle_c),
                                        orbital_filter = [s+'-'+self.orbs[0] for s in self.spins])
        for t, g in itt.product([transf_d, transf_c], [self.gloc, self.g0, self.se]):
            t.transform_g(g, reblock = False)
        self.quantum_numbers = [self.h_int.n_tot(), self.h_int.s2_tot()]

    def _site_transf(self, angle_d, angle_c):
        transf = {}
        for i, orb in enumerate(self.orbs):
            for spin in self.spins:
                if i == 0:
                    transf[spin+'-'+orb] = self._site_transf_mat(angle_d)
                elif i == 1:
                    transf[spin+'-'+orb] = self._site_transf_mat(angle_c)
                else:
                    assert False, 'more than two orbitals?!'
        return transf

    def _site_transf_mat(self, angle):
        return np.matrix([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
