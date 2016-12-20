import numpy as np, itertools as itt

from bethe.setups.generic import CycleSetupGeneric
from bethe.operators.hubbard import PlaquetteMomentum
from bethe.schemes.cdmft import GLocal, SelfEnergy, WeissField
from bethe.tightbinding import SquarelatticeDispersion as LatticeDispersion
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
        disp = LatticeDispersion(clusterhopping, n_k)
        self.site_transf_mat = dict([(s, .5 * np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]])) for s in spins])
        sites = range(4)
        self.old_struct = [[s, sites] for s in spins]
        self.new_struct = new_struct = [[s+'-'+a, [0]] for s, a in itt.product(spins, momenta)]
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
        
        
