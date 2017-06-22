import numpy as np, itertools as itt
from scipy.linalg import expm, eigh

from bethe.setups.generic import CycleSetupGeneric
from bethe.operators.hubbard import Site, TriangleMomentum, PlaquetteMomentum, Triangle, TriangleAIAO, TriangleSpinOrbitCoupling
from bethe.schemes.bethe import GLocal, WeissField, SelfEnergy, GLocalAFM, WeissFieldAFM, GLocalWithOffdiagonals, WeissFieldAIAO, WeissFieldAFM
from bethe.transformation import MatrixTransformation


class SingleBetheSetup(CycleSetupGeneric):

    def __init__(self, beta, mu, u, t_bethe, w1 = None, w2 = None, n_mom = 3, n_iw = 1025):
        up = "up"
        dn = "dn"
        spins = [up, dn]
        sites = range(1)
        hubbard = Site(u)
        t_loc = {up: np.zeros([1, 1]), dn: np.zeros([1, 1])}
        blocknames = spins
        blocksizes = [len(sites), len(sites)]
        gf_struct = [[s, sites] for s in spins]
        self.h_int = hubbard.get_h_int()
        self.gloc = GLocal(t_bethe, t_loc, w1, w2, n_mom, blocknames, blocksizes, beta, n_iw)
        self.g0 = WeissField(blocknames, blocksizes, beta, n_iw)
        self.se = SelfEnergy(blocknames, blocksizes, beta, n_iw)
        self.mu = mu
        self.global_moves = {"spin-flip": {("up", 0): ("dn", 0), ("dn", 0): ("up", 0)}}
        self.quantum_numbers = [hubbard.get_n_tot(), hubbard.get_n_per_spin(up)]


class TriangleBetheSetup(CycleSetupGeneric):
    """
    Contributions by Kristina Klafka
    """
    def __init__(self, beta, mu, u, t_triangle, t_bethe, orbital_labels = ["E", "A2", "A1"],
                 symmetric_orbitals = ["A2", "A1"],
                 site_transformation = np.array([[1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],[0,-1/np.sqrt(2),1/np.sqrt(2)],[-np.sqrt(2./3.),1/np.sqrt(6),1/np.sqrt(6)]]),
                 n_iw = 1025, afm = False):
        up = "up"
        dn = "dn"
        spins = [up, dn]
        sites = range(3)
        blocknames = [s+"-"+k for s in spins for k in orbital_labels]
        blocksizes = [1] * len(blocknames)
        gf_struct = [[n, range(s)] for n, s in zip(blocknames, blocksizes)]
        gf_struct_site = [[s, sites] for s in spins]
        transfbmat = dict([(s, site_transformation) for s in spins])
        transf = self.transf = MatrixTransformation(gf_struct_site, transfbmat, gf_struct)
        a = t_triangle
        t_loc_per_spin = np.array([[0,a,a],[a,0,a],[a,a,0]])
        t_loc = {up: t_loc_per_spin, dn: t_loc_per_spin}
        t_loc = transf.transform_matrix(t_loc)
        hubbard = TriangleMomentum(u, spins, orbital_labels, transfbmat)
        xy = symmetric_orbitals
        self.h_int = hubbard
        if afm:
            self.g0 = WeissFieldAFM(blocknames, blocksizes, beta, n_iw)
        else:
            self.g0 = WeissField(blocknames, blocksizes, beta, n_iw)
        self.gloc = GLocalWithOffdiagonals(t_bethe, t_loc, blocknames, blocksizes, beta, n_iw)
        self.se = SelfEnergy(blocknames, blocksizes, beta, n_iw)
        self.mu = mu
        self.global_moves = {}#{"spin-flip": dict([((s1+"-"+k, 0), (s2+"-"+k, 0)) for k in orbital_labels for s1, s2 in itt.product(spins, spins) if s1 != s2]), "A1A2-flip": dict([((s+"-"+k1, 0), (s+"-"+k2, 0)) for s in spins for k1, k2 in itt.product(xy, xy) if k1 != k2])}
        self.quantum_numbers = [hubbard.get_n_tot(), hubbard.get_n_per_spin(up)]


class TriangleAIAOBetheSetup(CycleSetupGeneric):
    """
    merges spin and sitespaces and diagonalizes(?)
    space hierarchy: spin, site
    """
    def __init__(self, beta, mu, u , t_triangle, t_bethe, n_iw = 1025, force_real = True,
                 site_transformation = np.array([[1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],[0,-1/np.sqrt(2),1/np.sqrt(2)],[-np.sqrt(2./3.),1/np.sqrt(6),1/np.sqrt(6)]])):
        self.site_transf = site_transformation
        sites = range(3)
        spins = ['up', 'dn']
        gf_struct = [['spin-mom', range(6)]]
        gf_struct_site = [[s, range(3)] for s in spins]
        blocknames = ['spin-mom']
        blocksizes = [len(sites)*2]
        t = t_triangle
        hop = np.array([[0,t,t],[t,0,t],[t,t,0]])
        t_loc = {s: hop for s in spins}
        site_transformation = {s: site_transformation for s in spins}
        transf = MatrixTransformation(gf_struct_site, site_transformation, gf_struct)
        self.t_loc = transf.transform_matrix(t_loc)
        self.h_int = TriangleSpinOrbitCoupling('spin-mom', u, transf = site_transformation)
        self.mu = mu
        self.gloc = GLocalWithOffdiagonals(t_bethe, self.t_loc, blocknames, blocksizes, beta, n_iw)
        self.se = SelfEnergy(blocknames, blocksizes, beta, n_iw)
        self.g0 = WeissFieldAIAO(blocknames, blocksizes, beta, n_iw)
        self.global_moves = {}#{"spin-flip": dict([((s1, i), (s2, i)) for i in sites for s1, s2 in itt.product(spins, spins) if s1 != s2])}
        self.quantum_numbers = [self.h_int.n_tot()]

    def rotate_t_loc(self, t_loc, force_real = False):
        t_loc_new = {'spin-site': np.zeros([6, 6], dtype = complex)}
        for a, b in itt.product(*[range(6)]*2):
            i, j = self.site_index(a), self.site_index(b)
            t1, t2 = self.spin_index(a), self.spin_index(b)
            #t_loc_new['spin-site'][a, b] += np.sum([self.spin_transf_mat(np.pi*.5, i*2*np.pi/3.).conjugate().transpose()[t1, s1] * t_loc['spin-site'][s1*3+i, s2*3+j] * self.spin_transf_mat(np.pi*.5, j*2*np.pi/3.)[s2, t2] for s1, s2 in itt.product(*[range(2)]*2)])
            t_loc_new['spin-site'][a, b] += np.sum([self.spin_transf_mat(i*2*np.pi/3., 0).conjugate().transpose()[t1, s1] * t_loc['spin-site'][s1*3+i, s2*3+j] * self.spin_transf_mat(j*2*np.pi/3., 0)[s2, t2] for s1, s2 in itt.product(*[range(2)]*2)])
        if force_real:
            tmp = np.empty([6]*2)
            tmp[:,:] = t_loc_new['spin-site'].real
            t_loc_new['spin-site'] = tmp
        return t_loc_new

    def superindex(self, spin_index, site_index):
        return spin_index * 3 + site_index

    def spin_index(self, superindex):
        return superindex / 3

    def site_index(self, superindex):
        return superindex % 3

    def spin_transf_mat(self, theta, phi = 0):
        py = np.matrix([[0,complex(0,-1)],[complex(0,1),0]])
        pz = np.matrix([[1,0],[0,-1]])
        return expm(complex(0,-1)*theta*py*.5)#.dot(expm(complex(0,1)*phi*pz*.5))


class PlaquetteBetheSetup(CycleSetupGeneric):
    """
    site transformation must be unitary and diagonalize G, assuming all sites are equal
    """
    def __init__(self, beta, mu, u, tnn_plaquette, tnnn_plaquette, t_bethe, w1 = None, w2 = None,
                 n_mom = 3, orbital_labels = ["G", "X", "Y", "M"], symmetric_orbitals = ["X", "Y"],
                 site_transformation =.5*np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]),
                 n_iw = 1025):
        up = "up"
        dn = "dn"
        spins = [up, dn]
        sites = range(4)
        blocknames = [s+"-"+k for s in spins for k in orbital_labels]
        blocksizes = [1] * len(blocknames)
        gf_struct = [[n, range(s)] for n, s in zip(blocknames, blocksizes)]
        gf_struct_site = [[s, sites] for s in spins]
        transfbmat = dict([(s, site_transformation) for s in spins])
        transf = MatrixTransformation(gf_struct_site, transfbmat, gf_struct)
        a, b = tnn_plaquette, tnnn_plaquette
        t_loc_per_spin = np.array([[0,a,a,b],[a,0,b,a],[a,b,0,a],[b,a,a,0]])
        t_loc = {up: t_loc_per_spin, dn: t_loc_per_spin}
        t_loc = transf.transform_matrix(t_loc)
        hubbard = PlaquetteMomentum(u, spins, orbital_labels, transfbmat)
        xy = symmetric_orbitals
        self.h_int = hubbard.get_h_int()
        self.gloc = GLocal(t_bethe, t_loc, w1, w2, n_mom, blocknames, blocksizes, beta, n_iw)
        self.g0 = WeissField(blocknames, blocksizes, beta, n_iw)
        self.se = SelfEnergy(blocknames, blocksizes, beta, n_iw)
        self.mu = mu
        self.global_moves = {"spin-flip": dict([((s1+"-"+k, 0), (s2+"-"+k, 0)) for k in orbital_labels for s1, s2 in itt.product(spins, spins) if s1 != s2]), "XY-flip": dict([((s+"-"+k1, 0), (s+"-"+k2, 0)) for s in spins for k1, k2 in itt.product(xy, xy) if k1 != k2])}
        self.quantum_numbers = [hubbard.get_n_tot(), hubbard.get_n_per_spin(up)]

    def init_noninteracting(self):
        self.se.zero()

    def init_centered_semicirculars(self): # TODO test
        for n, b in self.se:
            b << self.gloc.make_matrix(self.mu)[n]


class NambuMomentumPlaquette: # TODO

    def __init__(self, beta, mu, u, tnn_plaquette, tnnn_plaquette, t_bethe = 1, n_iw = 1025):
        Bethe.__init__(self, beta, mu, u, t_bethe, n_iw)
        g = "G"
        x = "X"
        y = "Y"
        m = "M"
        up = "up"
        dn = "dn"
        self.spins = [up, dn]
        self.sites = range(4)
        self.momenta = [g, x, y, m]
        self.spinors = range(2)
        self.block_labels = [k for k in self.momenta]
        self.gf_struct = [[l, self.spinors] for l in self.block_labels]
        self.gf_struct_site = [[s, self.sites] for s in self.spins]
        transformation_matrix = .5 * np.array([[1,1,1,1],
                                               [1,-1,1,-1],
                                               [1,1,-1,-1],
                                               [1,-1,-1,1]])
        self.transformation = dict([(s, transformation_matrix) for s in self.spins])
        mom_transf = MatrixTransformation(self.gf_struct_site, self.transformation, self.gf_struct)
        a = tnn_plaquette
        b = tnnn_plaquette
        t_loc = np.array([[0,a,a,b],[a,0,b,a],[a,b,0,a],[b,a,a,0]])
        t_loc = {up: np.array(t_loc), dn: np.array(t_loc)}
        reblock_map = {(up,0,0):(g,0,0), (dn,0,0):(g,1,1), (up,1,1):(x,0,0), (dn,1,1):(x,1,1), (up,2,2):(y,0,0), (dn,2,2):(y,1,1), (up,3,3):(m,0,0), (dn,3,3):(m,1,1)}
        self.t_loc = mom_transf.reblock_by_map(mom_transf.transform_matrix(t_loc), reblock_map)
        mu = {up: mu * np.identity(4), dn: mu * np.identity(4)}
        self.mu = mom_transf.reblock_by_map(mom_transf.transform_matrix(mu), reblock_map)
        self.operators = HubbardPlaquetteMomentumNambu(u, self.spins, self.momenta, self.transformation)
        self.h_int = self.operators.get_h_int()
        self.g0 = WeissFieldNambu(self.momenta, [self.spinors]*4, self.beta, n_iw, self.t, self.t_loc)
        self.initial_g = GLocalNambu(self.momenta, [self.spinors]*4, self.beta, n_iw, self.t, self.t_loc, self.g0)
        self.initial_se = GLocal(self.momenta, [self.spinors]*4, self.beta, n_iw, self.t, self.t_loc)

    def set_initial_guess(self, selfenergy, g0, anom_field_factor = 0, transform = True):
        """initializes by previous non-nambu solution and anomalous field or by 
        nambu-solution"""
        if transform:
            self._transform_particlehole(selfenergy, self.initial_se)
            self._transform_particlehole(g0, self.g0)
            self._set_anomalous(anom_field_factor)
        else:
            self.initial_se.set_gf(selfenergy)
            self.g0.set_gf(g0)

    def _set_anomalous(self, factor):
        """d-wave, singlet"""
        xi = self.momenta[1]
        yi = self.momenta[2]
        g = self.initial_se.gf
        n_points = len([iwn for iwn in g.mesh])/2
        for offdiag in [[0,1], [1,0]]:
            for n  in [n_points, n_points-1]:
                inds = tuple([n] + offdiag)
                g[xi].data[inds] = factor * g.beta * .5
            offdiag = tuple(offdiag)
            g[yi][offdiag] << -1 * g[xi][offdiag]

    def _transform_particlehole(self, g_sm, g_nambu):
        """gets a non-nambu greensfunction to initialize nambu"""
        gf_struct_mom = dict([(s+'-'+k, [0]) for s in self.spins for k in self.momenta])
        to_nambu = MatrixTransformation(gf_struct_mom, None, self.gf_struct)
        up, dn = self.spins
        reblock_map = [[(up+'-'+k,0,0), (k,0,0)] for k in self.momenta]
        reblock_map += [[(dn+'-'+k,0,0), (k,1,1)]  for k in self.momenta]
        reblock_map = dict(reblock_map)
        for b, i, j in g_sm.all_indices:
            b_nam, i_nam, j_nam = reblock_map[(b, int(i), int(j))]
            if i_nam == 0 and j_nam == 0:
                g_nambu.gf[b_nam][i_nam, j_nam] << g_sm[b][i, j]
            if i_nam == 1 and j_nam == 1:
                g_nambu.gf[b_nam][i_nam, j_nam] << -1 * g_sm[b][i, j].conjugate()
