import numpy as np
from pytriqs.gf.local import BlockGf, GfImFreq, iOmega_n, inverse

from greensfunctions import MatsubaraGreensFunction
from gfoperations import double_dot_product


class WeissField(MatsubaraGreensFunction):
    """
    RevModPhys.68.13 Eq. (23) generalized to a cluster-impurity, i.e. quantities
    becoming matrices over the clustersites.
    """
    def __init__(self, name_list, block_states, beta, n_iw, t, t_loc):
        MatsubaraGreensFunction.__init__(self, name_list, block_states, beta, n_iw)
        self.t = t
        self.t_loc = t_loc

    def zero(self):
        return dict([[bn, np.zeros([len(bs)] * 2)] for bn, bs in zip(self.block_names, self.block_states)])

    def calc_selfconsistency(self, gf_local, mu = None, mu_number = None):
        assert (mu is None) ^ (mu_number is None), "Self-consistency condition depends on either mu or mu_number"
        if not mu_number is None:
            self.set_mu(mu_number)
        if not mu is None:
            self.mu = mu
        for bn, b in self.gf:
            b << inverse(iOmega_n  + self.mu[bn] - self.t_loc[bn] - self.t**2 * gf_local[bn])

    def set_mu(self, mu_number):
        self.mu = dict([[bn, mu_number * np.identity(len(bs))] for bn, bs in zip(self.block_names, self.block_states)])
