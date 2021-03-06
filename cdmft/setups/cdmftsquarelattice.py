import numpy as np, itertools as itt

from cdmft.setups.common import CycleSetupCommon
from cdmft.setups.bethelattice import NambuMomentumPlaquette as BetheNambuMomentumPlaquette
from cdmft.operators.hubbard import PlaquetteMomentum, PlaquetteMomentumNambu
from cdmft.schemes.cdmft import GLocal, SelfEnergy, WeissField, GLocalNambu, WeissFieldNambu
from cdmft.tightbinding import SquarelatticeDispersion, LatticeDispersion
from cdmft.transformation2 import Transformation, Reblock, UnitaryMatrixTransformation


def get_hopping(lattice, tnn, tnnn, tz, alpha_x = 1, alpha_y = 1, alpha_prime = 1):
    t, s = tnn, tnnn
    tax, tay, sap = alpha_x*t, alpha_y*t, alpha_prime*s
    assert lattice in ['2d','3d','3dandersen'], 'lattice unkown'
    if lattice == '2d':
        clusterhopping = {(0,0):[[0,t,t,s],[t,0,s,t],[t,s,0,t],[s,t,t,0]],
                          (1,0):[[0,tax,0,sap],[0,0,0,0],[0,sap,0,tax],[0,0,0,0]],
                          (1,1):[[0,0,0,sap],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                          (0,1):[[0,0,tay,sap],[0,0,sap,tay],[0,0,0,0],[0,0,0,0]],
                          (-1,1):[[0,0,0,0],[0,0,sap,0],[0,0,0,0],[0,0,0,0]],
                          (-1,0):[[0,0,0,0],[tax,0,sap,0,],[0,0,0,0],[sap,0,tax,0]],
                          (-1,-1):[[0,0,0,0],[0,0,0,0],[0,0,0,0],[sap,0,0,0]],
                          (0,-1):[[0,0,0,0],[0,0,0,0],[tay,sap,0,0],[sap,tay,0,0]],
                          (1,-1):[[0,0,0,0],[0,0,0,0],[0,sap,0,0],[0,0,0,0]]}
    elif lattice == '3d':
        clusterhopping = {(0,0,0): [[0,t,t,s],[t,0,s,t],[t,s,0,t],[s,t,t,0]],
                          (1,0,0): [[0,t,0,s],[0,0,0,0],[0,s,0,t],[0,0,0,0]],
                          (1,1,0): [[0,0,0,s],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                          (0,1,0): [[0,0,t,s],[0,0,s,t],[0,0,0,0],[0,0,0,0]],
                          (-1,1,0): [[0,0,0,0],[0,0,s,0],[0,0,0,0],[0,0,0,0]],
                          (-1,0,0): [[0,0,0,0],[t,0,s,0,],[0,0,0,0],[s,0,t,0]],
                          (-1,-1,0): [[0,0,0,0],[0,0,0,0],[0,0,0,0],[s,0,0,0]],
                          (0,-1,0): [[0,0,0,0],[0,0,0,0],[t,s,0,0],[s,t,0,0]],
                          (1,-1,0): [[0,0,0,0],[0,0,0,0],[0,s,0,0],[0,0,0,0]],
                          (0,0,1): [[tz,0,0,0],[0,tz,0,0],[0,0,tz,0],[0,0,0,tz]],
                          (0,0,-1): [[tz,0,0,0],[0,tz,0,0],[0,0,tz,0],[0,0,0,tz]]}
    elif lattice == '3dandersen':
        u,v,w = tz, tz*.25, -tz*.5
        clusterhopping = {(0,0,0): [[0,t,t,s],[t,0,s,t],[t,s,0,t],[s,t,t,0]],
                          (1,0,0): [[0,t,0,s],[0,0,0,0],[0,s,0,t],[0,0,0,0]],
                          (1,1,0): [[0,0,0,s],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                          (0,1,0): [[0,0,t,s],[0,0,s,t],[0,0,0,0],[0,0,0,0]],
                          (-1,1,0): [[0,0,0,0],[0,0,s,0],[0,0,0,0],[0,0,0,0]],
                          (-1,0,0): [[0,0,0,0],[t,0,s,0,],[0,0,0,0],[s,0,t,0]],
                          (-1,-1,0): [[0,0,0,0],[0,0,0,0],[0,0,0,0],[s,0,0,0]],
                          (0,-1,0): [[0,0,0,0],[0,0,0,0],[t,s,0,0],[s,t,0,0]],
                          (1,-1,0): [[0,0,0,0],[0,0,0,0],[0,s,0,0],[0,0,0,0]],

                          (0,0,1): [[u,0,0,w],[0,u,w,0],[0,w,u,0],[w,0,0,u]],
                          (0,0,-1): [[u,0,0,w],[0,u,w,0],[0,w,u,0],[w,0,0,u]],

                          (1,1,1): [[0,0,0,w],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                          (1,1,-1): [[0,0,0,w],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                          (-1,1,1): [[0,0,0,0],[0,0,w,0],[0,0,0,0],[0,0,0,0]],
                          (-1,1,-1): [[0,0,0,0],[0,0,w,0],[0,0,0,0],[0,0,0,0]],
                          (-1,-1,1): [[0,0,0,0],[0,0,0,0],[0,0,0,0],[w,0,0,0]],
                          (-1,-1,-1) :[[0,0,0,0],[0,0,0,0],[0,0,0,0],[w,0,0,0]],
                          (1,-1,1): [[0,0,0,0],[0,0,0,0],[0,w,0,0],[0,0,0,0]],
                          (1,-1,-1): [[0,0,0,0],[0,0,0,0],[0,w,0,0],[0,0,0,0]],

                          (1,0,-1): [[v,0,0,w],[0,v,0,0],[0,w,v,0],[0,0,0,v]],
                          (1,0,1): [[v,0,0,w],[0,v,0,0],[0,w,v,0],[0,0,0,v]],
                          (0,1,-1): [[v,0,0,w],[0,v,w,0],[0,0,v,0],[0,0,0,v]],
                          (0,1,1): [[v,0,0,w],[0,v,w,0],[0,0,v,0],[0,0,0,v]],
                          (-1,0,-1): [[v,0,0,0],[0,v,w,0],[0,0,v,0],[w,0,0,v]],
                          (-1,0,1): [[v,0,0,0],[0,v,w,0],[0,0,v,0],[w,0,0,v]],
                          (0,-1,-1): [[v,0,0,0],[0,v,0,0],[0,w,v,0],[w,0,0,v]],
                          (0,-1,1): [[v,0,0,0],[0,v,0,0],[0,w,v,0],[w,0,0,v]]}
    return clusterhopping


class MomentumPlaquetteSetup(CycleSetupCommon):
    """
    Plaquette(2by2)-cluster of the 2D squarelattice within Cluster DMFT
    assumes that transformation_matrix diagonalizes the GLocal on site-space
    i.e. the four clustersites are equivalent, no broken symmetry
    """
    def __init__(self, beta, mu, u, tnn, tnnn, n_k, spins = ['up', 'dn'], momenta = ['G', 'X', 'Y', 'M'], equivalent_momenta = ['X', 'Y'], n_iw = 1025, tightbinding = '2d', tz = -.15):
        self.spins = spins
        transf_mat = dict([(s, .5 * np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]])) for s in spins])
        mom_struct = [[s+'-'+a, [0]] for s, a in itt.product(spins, momenta)]
        site_struct = [[s, range(4)] for s in spins]
        up, dn, a, b, c, d = tuple(spins + momenta)
        reblock_map = {(up,0,0): (up+'-'+a,0,0), (up,1,1): (up+'-'+b,0,0),
                       (up,2,2): (up+'-'+c,0,0), (up,3,3): (up+'-'+d,0,0),
                       (dn,0,0): (dn+'-'+a,0,0), (dn,1,1): (dn+'-'+b,0,0),
                       (dn,2,2): (dn+'-'+c,0,0), (dn,3,3): (dn+'-'+d,0,0)}
        inverse_reblock_map = {val: key for key, val in reblock_map.items()}
        ksum_unblock = Transformation([Reblock(site_struct, mom_struct, inverse_reblock_map)])
        self.momentum_transf = Transformation([UnitaryMatrixTransformation(transf_mat)])

        clusterhopping = get_hopping(tightbinding, tnn, tnnn, tz)
        for r, t in clusterhopping.items():
            clusterhopping[r] = np.array(t)
        self.disp = LatticeDispersion(clusterhopping, n_k)
        self.disp.transform(self.momentum_transf)
        hubbard = PlaquetteMomentum(u, spins, momenta, transf_mat)
        self.h_int = hubbard.get_h_int()
        self.gloc = GLocal(self.disp, ksum_unblock, [s[0] for s in mom_struct], [1] * 8, beta, n_iw)
        self.g0 = WeissField([s[0] for s in mom_struct], [1] * 8, beta, n_iw)
        self.se = SelfEnergy([s[0] for s in mom_struct], [1] * 8, beta, n_iw)
        self.mu = mu
        xy = equivalent_momenta
        self.global_moves = {"spin-flip": dict([((s1+"-"+k, 0), (s2+"-"+k, 0)) for k in momenta for s1, s2 in itt.product(spins, spins) if s1 != s2]),
                             "XY-flip": dict([((s+"-"+k1, 0), (s+"-"+k2, 0)) for s in spins for k1, k2 in itt.product(xy, xy) if k1 != k2])}
        self.quantum_numbers = [hubbard.n_tot(), hubbard.sz_tot()]


class NambuMomentumPlaquetteSetup(BetheNambuMomentumPlaquette):
    """
    Don't use self.mom_transf for the backtransformation! The reblock algorithm will drop 
    off-diagonals
    """
    def __init__(self, beta, mu, u, tnn, tnnn, n_k, spins = ['up', 'dn'], momenta = ['G', 'X', 'Y', 'M'], n_iw = 1025, tightbinding = '2d', tz = -.15, alpha_x = 1, alpha_y = 1, alpha_prime = 1):
        g, x, y, m = "G", "X", "Y", "M"
        up, dn = "up", "dn"
        self.spins = [up, dn]
        self.sites = range(4)
        self.momenta = [g, x, y, m]
        self.spinors = range(2) #nambu spinors: 0: particle, 1: hole
        self.block_labels = [k for k in self.momenta]
        self.gf_struct = [[l, self.spinors] for l in self.block_labels]
        self.gf_struct_site = [[s, self.sites] for s in self.spins]
        self.gf_struct_tot = [['0', range(8)]]
        self.reblock_map = {(up,0,0):(g,0,0), (dn,0,0):(g,1,1), (up,1,1):(x,0,0), (dn,1,1):(x,1,1), (up,2,2):(y,0,0), (dn,2,2):(y,1,1), (up,3,3):(m,0,0), (dn,3,3):(m,1,1)}
        transformation_matrix = .5 * np.array([[1,1,1,1],
                                               [1,-1,1,-1],
                                               [1,1,-1,-1],
                                               [1,-1,-1,1]])
        self.transformation = dict([(s, transformation_matrix) for s in self.spins])
        #self.mom_transf = MatrixTransformation(self.gf_struct_site, self.transformation, self.gf_struct, reblock_map = self.reblock_map)
        
        disp_transf_mat =  {'0': np.kron(np.eye(2), transformation_matrix)}
        disp_transf = Transformation([UnitaryMatrixTransformation(disp_transf_mat)])
        
        k = {'G': 0, 'X': 1, 'Y': 2, 'M': 3}
        reblock_ksum_map = {(K,i,j): ('0', k[K]+i*4, k[K]+j*4) for K, i, j in itt.product(momenta, range(2), range(2))}
        self.reblock_ksum = Transformation([Reblock(self.gf_struct_tot, self.gf_struct, reblock_ksum_map)])
        
        clusterhopping = get_hopping(tightbinding, tnn, tnnn, tz, alpha_x, alpha_y, alpha_prime)
        for r, t in clusterhopping.items():
            clusterhopping[r] = np.kron(np.eye(2), np.array(t))
        self.disp = LatticeDispersion(clusterhopping, n_k, spins = ['0'])
        self.disp.transform(disp_transf)
        self.mu = mu
        self.operators = PlaquetteMomentumNambu(u, self.spins, self.momenta, self.transformation)
        self.h_int = self.operators.get_h_int()
        self.gloc = GLocalNambu(self.disp, self.reblock_ksum, self.momenta, [2]*4, beta, n_iw)
        self.g0 = WeissFieldNambu(self.momenta, [2]*4, beta, n_iw)
        self.se = SelfEnergy(self.momenta, [2]*4, beta, n_iw)
        self.global_moves = {"XY-flip": dict([((k1, i), (k2, i)) for i in range(2) for k1, k2 in itt.product([x,y], [x,y]) if k1 != k2])}
        self.quantum_numbers = []
