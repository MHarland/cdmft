import numpy as np
from pytriqs.gf.local import BlockGf, GfImFreq, SemiCircular

from hamiltonian import HubbardSite, HubbardPlaquette


class SingleSiteBethe:

    def __init__(self, beta, mu, u, t = 1, n_iw = 1025):
        self.dim = 1
        self.beta = beta
        up = "up"
        dn = "dn"
        self.gf_struct = [[up, range(1)], [dn, range(1)]]
        self.u = u
        self.t_loc = {up: np.zeros([1, 1]), dn: np.zeros([1, 1])}
        self.mu = {up: mu * np.identity(1), dn: mu * np.identity(1)}
        h = HubbardSite(u, [up, dn])
        self.h_int = h.get_h_int()
        self.t = t
        self.bandwidth = 4 * t
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
