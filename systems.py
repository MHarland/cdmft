import numpy as np
from pytriqs.gf.local import BlockGf, GfImFreq, SemiCircular

from hamiltonian import HubbardSite


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
        self.t = 1
        self.bandwidth = 4 * t
        self.initial_guess = BlockGf(name_list = [b[0] for b in self.gf_struct],
                                     block_list = [GfImFreq(n_points = n_iw, beta = beta, indices = b[1]) for b in self.gf_struct])
        for s, b in self.initial_guess:
            b << SemiCircular(self.bandwidth * .5)
