import numpy as np, itertools as itt

from bethe.setups.generic import CycleSetupGeneric
from bethe.setups.bethelattice import NambuMomentumPlaquette as BetheNambuMomentumPlaquette
from bethe.operators.hubbard import PlaquetteMomentum, PlaquetteMomentumNambu
from bethe.schemes.cdmft import GLocal, SelfEnergy, WeissField, GLocalNambu
from bethe.tightbinding import SquarelatticeDispersion, LatticeDispersion
from bethe.transformation import MatrixTransformation


class MomentumPlaquetteSetup(CycleSetupGeneric):
    """
    Plaquette(2by2)-cluster of the 2D squarelattice within Cluster DMFT
    assumes that transformation_matrix diagonalizes the GLocal on site-space
    i.e. the four clustersites are equivalent, no broken symmetry
    """
    def __init__(self, beta, mu, u, tnn, tnnn, n_k, spins = ['up', 'dn'], momenta = ['G', 'X', 'Y', 'M'], equivalent_momenta = ['X', 'Y'], n_iw = 1025):
        # TODO add angle(s) for degenerate site transformations
        t, s = tnn, tnnn
        clusterhopping = {(0,0):[[0,t,t,s],[t,0,s,t],[t,s,0,t],[s,t,t,0]],
                          (1,0):[[0,t,0,s],[0,0,0,0],[0,s,0,t],[0,0,0,0]],
                          (1,1):[[0,0,0,s],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                          (0,1):[[0,0,t,s],[0,0,s,t],[0,0,0,0],[0,0,0,0]],
                          (-1,1):[[0,0,0,0],[0,0,s,0],[0,0,0,0],[0,0,0,0]],
                          (-1,0):[[0,0,0,0],[t,0,s,0,],[0,0,0,0],[s,0,t,0]],
                          (-1,-1):[[0,0,0,0],[0,0,0,0],[0,0,0,0],[s,0,0,0]],
                          (0,-1):[[0,0,0,0],[0,0,0,0],[t,s,0,0],[s,t,0,0]],
                          (1,-1):[[0,0,0,0],[0,0,0,0],[0,s,0,0],[0,0,0,0]]}
        for r, t in clusterhopping.items():
            clusterhopping[r] = np.array(t)
        disp = SquarelatticeDispersion(clusterhopping, n_k)
        disp.transform_site_space(self.site_transf_mat, new_struct)#, reblock_map)
        hubbard = PlaquetteMomentum(u, spins, momenta, self.site_transf_mat)
        self.h_int = hubbard.get_h_int()
        self.gloc = GLocal(disp, [s[0] for s in new_struct], [1] * 8, beta, n_iw)
        self.g0 = WeissField([s[0] for s in new_struct], [1] * 8, beta, n_iw)
        self.se = SelfEnergy([s[0] for s in new_struct], [1] * 8, beta, n_iw)
        self.mu = mu
        xy = equivalent_momenta
        self.global_moves = {"spin-flip": dict([((s1+"-"+k, 0), (s2+"-"+k, 0)) for k in momenta for s1, s2 in itt.product(spins, spins) if s1 != s2]),
                             "XY-flip": dict([((s+"-"+k1, 0), (s+"-"+k2, 0)) for s in spins for k1, k2 in itt.product(xy, xy) if k1 != k2])}
        self.quantum_numbers = [hubbard.get_n_tot(), hubbard.get_n_per_spin(spins[0])]


class NambuMomentumPlaquetteSetup(BetheNambuMomentumPlaquette):
    """
    """
    def __init__(self, beta, mu, u, tnn, tnnn, n_k, spins = ['up', 'dn'], momenta = ['G', 'X', 'Y', 'M'], n_iw = 1025):
        g, x, y, m = "G", "X", "Y", "M"
        up, dn = "up", "dn"
        self.spins = [up, dn]
        self.sites = range(4)
        self.momenta = [g, x, y, m]
        self.spinors = range(2) #nambu spinors: 0: particle, 1: hole
        self.block_labels = [k for k in self.momenta]
        self.gf_struct = [[l, self.spinors] for l in self.block_labels]
        self.gf_struct_site = [[s, self.sites] for s in self.spins]
        self.reblock_map = {(up,0,0):(g,0,0), (dn,0,0):(g,1,1), (up,1,1):(x,0,0), (dn,1,1):(x,1,1), (up,2,2):(y,0,0), (dn,2,2):(y,1,1), (up,3,3):(m,0,0), (dn,3,3):(m,1,1)}
        transformation_matrix = .5 * np.array([[1,1,1,1],
                                               [1,-1,1,-1],
                                               [1,1,-1,-1],
                                               [1,-1,-1,1]])
        self.transformation = dict([(s, transformation_matrix) for s in self.spins])
        self.mom_transf = MatrixTransformation(self.gf_struct_site, self.transformation, self.gf_struct, reblock_map = self.reblock_map)
        t, s = tnn, tnnn
        clusterhopping = {(0,0):[[0,t,t,s],[t,0,s,t],[t,s,0,t],[s,t,t,0]],
                          (1,0):[[0,t,0,s],[0,0,0,0],[0,s,0,t],[0,0,0,0]],
                          (1,1):[[0,0,0,s],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                          (0,1):[[0,0,t,s],[0,0,s,t],[0,0,0,0],[0,0,0,0]],
                          (-1,1):[[0,0,0,0],[0,0,s,0],[0,0,0,0],[0,0,0,0]],
                          (-1,0):[[0,0,0,0],[t,0,s,0,],[0,0,0,0],[s,0,t,0]],
                          (-1,-1):[[0,0,0,0],[0,0,0,0],[0,0,0,0],[s,0,0,0]],
                          (0,-1):[[0,0,0,0],[0,0,0,0],[t,s,0,0],[s,t,0,0]],
                          (1,-1):[[0,0,0,0],[0,0,0,0],[0,s,0,0],[0,0,0,0]]}
        for r, t in clusterhopping.items():
            clusterhopping[r] = np.array(t)
        disp = LatticeDispersion(clusterhopping, n_k)
        disp.transform(self.mom_transf)
        self.mu = mu
        self.operators = PlaquetteMomentumNambu(u, self.spins, self.momenta, self.transformation)
        self.h_int = self.operators.get_h_int()
        self.gloc = GLocalNambu(disp, self.momenta, [2]*4, beta, n_iw)
        self.g0 = WeissField(self.momenta, [2]*4, beta, n_iw)
        self.se = SelfEnergy(self.momenta, [2]*4, beta, n_iw)
        self.global_moves = {}
        self.quantum_numbers = []
