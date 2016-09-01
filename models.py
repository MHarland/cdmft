import numpy as np, itertools as itt
from pytriqs.gf.local import BlockGf, GfImFreq, SemiCircular, iOmega_n, inverse
from pytriqs.gf.local.descriptor_base import Function

from glocal import GLocal
from hamiltonian import HubbardSite, HubbardPlaquette, HubbardPlaquetteMomentum, HubbardPlaquetteMomentumNambu
from transformation import MatrixTransformation, InterfaceToBlockstructure

class Bethe:

    def __init__(self, beta, mu, u, t_bethe = 1, n_iw = 1025):
        self.beta = beta
        self.mu = float(mu)
        self.u = u
        self.t = t_bethe
        self.n_iw = n_iw
        self.bandwidth = 4 * t_bethe

    def init_dmft(self):
        return {"h_int": self.h_int, "g_local": self.initial_guess, "mu": self.mu}

class SingleSite(Bethe):

    def __init__(self, beta, mu, u, t_bethe = 1, n_iw = 1025):
        Bethe.__init__(self, beta, mu, u, t_bethe = 1, n_iw = 1025)
        up = "up"
        dn = "dn"
        self.gf_struct = [[up, range(1)], [dn, range(1)]]
        self.t_loc = {up: np.zeros([1, 1]), dn: np.zeros([1, 1])}
        h = HubbardSite(u, [up, dn])
        self.h_int = h.get_h_int()
        self.initial_guess = BlockGf(name_list = [b[0] for b in self.gf_struct],
                                     block_list = [GfImFreq(n_points = n_iw, beta = beta, indices = b[1]) for b in self.gf_struct])
        for s, b in self.initial_guess:
            b << SemiCircular(self.bandwidth * .5)


class PlaquetteBethe:

    def __init__(self, beta, mu, u, tnn_plaquette, tnnn_plaquette, t = 1, n_iw = 1025):
        self.dim = 2
        self.beta = beta
        up = "up"
        dn = "dn"
        self.gf_struct = [[up, range(4)], [dn, range(4)]]
        self.u = u
        a = tnn_plaquette
        b = tnnn_plaquette
        t_loc = [[0,a,a,b],[a,0,b,a],[a,b,0,a],[b,a,a,0]]
        self.t_loc = {up: np.array(t_loc), dn: np.array(t_loc)}
        self.mu = {up: mu * np.identity(4), dn: mu * np.identity(4)}
        h = HubbardPlaquette(u, [up, dn])
        self.h_int = h.get_h_int()
        self.t = t
        self.bandwidth = 4 * t
        self.initial_guess = BlockGf(name_list = [b[0] for b in self.gf_struct],
                                     block_list = [GfImFreq(n_points = n_iw, beta = beta, indices = b[1]) for b in self.gf_struct])
        for s, b in self.initial_guess:
            b << SemiCircular(self.bandwidth * .5)


class MomentumPlaquetteBethe:

    def __init__(self, beta, mu, u, tnn_plaquette, tnnn_plaquette, t = 1, n_iw = 1025):
        self.dim = 2
        self.beta = beta
        self.momenta = ["G", "X", "Y", "M"]
        up = "up"
        dn = "dn"
        self.spins = [up, dn]
        self.mu_number = mu
        self.sites = range(4)
        self.block_labels = [spin+"-"+k for spin in self.spins for k in self.momenta]
        self.gf_struct = [[l, range(1)] for l in self.block_labels]
        self.gf_struct_site = [[s, self.sites] for s in self.spins]
        self.u = u
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
        self.t_loc = mom_transf.reblock(mom_transf.transform_matrix(t_loc))
        mu = {up: mu * np.identity(4), dn: mu * np.identity(4)}
        self.mu = mom_transf.reblock(mom_transf.transform_matrix(mu))
        h = HubbardPlaquetteMomentum(u, self.spins, self.momenta, self.transformation)
        self.h_int = h.get_h_int()
        self.t = t
        self.bandwidth = 4 * t
        self.initial_guess = BlockGf(name_list = [b[0] for b in self.gf_struct],
                                     block_list = [GfImFreq(n_points = n_iw, beta = beta, indices = b[1]) for b in self.gf_struct])
        self.init_centered_semicirculars()

    def init_noninteracting(self):
        for sk, b in self.initial_guess:
            z = lambda iw: iw + (self.mu[sk] - self.t_loc[sk])[0, 0]
            gf = lambda iw: (z(iw) - complex(0, 1) * np.sign(z(iw).imag) * np.sqrt(4*self.t**2 - z(iw)**2))/(2*self.t**2)
            one = np.identity(b.N1)
            b.tail.zero()
            b.tail[1][:,:] = 1. * one
            b.tail[3][:,:] = self.t**2 * one
            b.tail[5][:,:] = 2 * self.t**4 * one
            b.tail.mask.fill(6)
            Function(gf, None)(b)

    def init_centered_semicirculars(self):
        for sk, b in self.initial_guess:
            b << SemiCircular(self.bandwidth * .5)


class NambuMomentumPlaquetteBethe:

    def __init__(self, beta, mu, u, tnn_plaquette, tnnn_plaquette, t = 1, n_iw = 1025):
        g = "G"
        x = "X"
        y = "Y"
        m = "M"
        up = "up"
        dn = "dn"
        self.beta = beta
        self.spins = [up, dn]
        self.sites = range(4)
        self.momenta = [g, x, y, m]
        self.spinors = range(2)
        self.block_labels = [k for k in self.momenta]
        self.gf_struct = [[l, self.spinors] for l in self.block_labels]
        self.gf_struct_site = [[s, self.sites] for s in self.spins]
        self.u = u
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
        h = HubbardPlaquetteMomentumNambu(u, self.spins, self.momenta, self.transformation)
        self.h_int = h.get_h_int()
        self.t = t
        self.bandwidth = 4 * t
        #self.initial_guess = BlockGf(name_list = [b[0] for b in self.gf_struct], block_list = [GfImFreq(n_points = n_iw, beta = beta, indices = b[1]) for b in self.gf_struct])
        self.initial_guess = GLocal(self.momenta, [self.spinors]*4, self.beta, n_iw)

    def init_guess(self, g_momentumplaquettebethe = None, anom_field_factor = None,
                   g_nambumomentumplaquettebethe = None):
        """initializes by previous non-nambu solution and anomalous field or by 
        nambu-solution"""
        if g_nambumomentumplaquettebethe is None:
            self._set_particlehole(g_momentumplaquettebethe)
            self._set_anomalous(anom_field_factor)
        else:
            self.initial_guess = g_nambumomentumplaquettebethe.copy()

    def _set_anomalous(self, factor):
        """d-wave, singlet"""
        xi = self.momenta[1]
        yi = self.momenta[2]
        g = self.initial_guess.gf
        n_points = len([iwn for iwn in g.mesh])/2
        for offdiag in [[0,1], [1,0]]:
            for n in [n_points, n_points-1]:
                inds = tuple([n] + offdiag)
                g[xi].data[inds] = factor * g.beta * .5
            offdiag = tuple(offdiag)
            g[yi][offdiag] << -1 * g[xi][offdiag]

    def _set_particlehole(self, g_mpb):
        """gets a non-nambu greensfunction to initialize nambu"""
        gf_struct_mom = dict([(s+k, [0]) for s in self.spins for k in self.momenta])
        to_nambu = MatrixTransformation(gf_struct_mom, None, self.gf_struct)
        up, dn = self.spins
        #g_nambuspace = InterfaceToBlockstructure(g_mpb, gf_struct_mom, self.gf_struct)
        reblock_map = [[(up+'-'+k,0,0), (k,0,0)] for k in self.momenta]
        reblock_map += [[(dn+'-'+k,0,0), (k,1,1)]  for k in self.momenta]
        reblock_map = dict(reblock_map)
        g_mpb = to_nambu.reblock_by_map(g_mpb, reblock_map)
        for block in self.gf_struct:
            blockname, blockindices = block
            self.initial_guess.gf[blockname][0, 0] << g_mpb[blockname][0, 0]
            self.initial_guess.gf[blockname][1, 1] << -1 * g_mpb[blockname][1, 1].conjugate()

    def _set_particlehole_noninteracting(self):
        """sets up the exact solution for U=0"""
        pauli3 = np.array([[1,0],[0,-1]])
