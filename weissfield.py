import numpy as np
from pytriqs.gf.local import BlockGf, GfImFreq, iOmega_n, inverse

from greensfunctions import MatsubaraGreensFunction
from gfoperations import double_dot_product

class WeissField(MatsubaraGreensFunction):
    
    """RevModPhys.68.13 Eq. (23) generalized to a cluster-impurity, i.e. quantities becoming matrices over the clustersites."""

    def __init__(self, name_list, block_states, beta, n_iw, t, mu = None, t_loc = None):
        MatsubaraGreensFunction.__init__(self, name_list, block_states, beta, n_iw)
        self.t = t
        if mu is None:
            self.mu = self.zero()
        else:
            self.mu = mu
        if t_loc is None:
            self.t_loc = self.zero()
        else:
            self.t_loc = t_loc

    def zero(self):
        return dict([[bn, np.zeros([len(bs)] * 2)] for bn, bs in zip(self.block_names, self.block_states)])

    def calc_selfconsistency(self, gf_local, mu_number = None):
        if not mu_number is None:
            self.set_mu(mu_number)
        for bn, b in self.gf:
            b << inverse(iOmega_n  + self.mu[bn] + self.t_loc[bn] -
                         double_dot_product(self.t[bn], gf_local[bn], self.t[bn]))

    def set_mu(self, mu_number):
        self.mu = dict([[bn, mu_number * np.identity(len(bs))] for bn, bs in zip(self.block_names, self.block_states)])
