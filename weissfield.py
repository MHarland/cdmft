import numpy as np, itertools as itt
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


class WeissFieldNambu(WeissField):
    """
    RevModPhys.68.13 Eq. (101)
    Modified self-consistency condition since the particle-hole transformation on 
    spin-down is not unitary, gf_struct is considered to be of 2x2 Nambu-blocks
    """
    def __init__(self, *args, **kwargs):
        WeissField.__init__(self, *args, **kwargs)
        self.anom_field = None
        
    def calc_selfconsistency(self, gf_local, mu = None):
        for name, block in gf_local:
            assert len(block.data[0,:,:]) == 2, "gf_struct not with implemented Nambu self-consistency compatible"
        one = np.identity(2)
        pauli3 = np.array([[1,0],[0,-1]])
        if not mu is None:
            self.mu = mu
        for bn, b in self.gf:
            if self.anom_field is None:
                tmp1 = pauli3.dot(self.mu[bn] - self.t_loc[bn])
            else:
                tmp1 = pauli3.dot(self.mu[bn] + self.anom_field[bn] - self.t_loc[bn])
            tmp2 = double_dot_product(pauli3, gf_local[bn], pauli3)
            b << inverse(iOmega_n  + tmp1 - self.t**2 * tmp2)

    def set_anomalous_dwave_field(self, field):
        if field == None:
            self.anom_field = None
        else:
            self.anom_field = {}
            for block in self.gf.indices:
                self.anom_field[block] = np.zeros([2, 2])
            for block, matrix in field.items():
                self.anom_field[block] = matrix
