import numpy as np, itertools as itt
from pytriqs.gf.local import BlockGf, GfImFreq, iOmega_n, inverse, GfImTime

from greensfunctions import MatsubaraGreensFunction
from gfoperations import double_dot_product


class WeissField(MatsubaraGreensFunction):
    """
    RevModPhys.68.13 Eq. (23) generalized to a cluster-impurity, i.e. quantities
    becoming matrices over the clustersites.
    """
    def __init__(self, block_names, block_states, beta, n_iw, t, t_loc):
        MatsubaraGreensFunction.__init__(self, block_names, block_states, beta, n_iw)
        self.t = t
        self.t_loc = t_loc

    def calc_selfconsistency(self, g_loc, mu):
        if isinstance(mu, float):
            mu = self._get_mu_matrix(mu)
        for bn, b in self.gf:
            b << inverse(iOmega_n  + mu[bn] - self.t_loc[bn] - self.t**2 * g_loc.gf[bn])

    def _get_mu_matrix(self, mu_number):
        return dict([[bn, mu_number * np.identity(len(bs))] for bn, bs in zip(self.block_names, self.block_states)])

    def make_g_tau_real(self, n_tau):
        """Transforms to tau space with n_tau meshsize, sets self.gf accordingly"""
        inds_tau = range(n_tau)
        g_tau = BlockGf(name_list = self.block_names,
                         block_list = [GfImTime(beta = self.beta, indices = s,
                                                n_points = n_tau) for s in self.block_states])
        for bname, b in g_tau:
            b.set_from_inverse_fourier(self.gf[bname])
            inds_block = range(len(b.data[0,:,:]))
            for n, i, j in itt.product(inds_tau, inds_block, inds_block):
                b.data[n,i,j] = b.data[n,i,j].real
            self.gf[bname].set_from_fourier(b)


class WeissFieldNambu(WeissField):
    """
    RevModPhys.68.13 Eq. (101)
    Modified self-consistency condition since the particle-hole transformation on 
    spin-down is not unitary, gf_struct is considered to be of 2x2 Nambu-blocks
    broken_symmetry_map maps within the self-consistency equation the Weissfield
    index to the corresponding local Greensfunction index
    """
    def __init__(self, name_list, block_states, beta, n_iw, t, t_loc,
                 broken_symmetry_map = {"G": "G", "X": "X", "Y": "Y", "M": "M"}):
        WeissField.__init__(self, name_list, block_states, beta, n_iw, t, t_loc)
        self.bs_map = broken_symmetry_map

    def calc_selfconsistency(self, g_loc, mu):
        for name, block in gf_local:
            assert len(block.data[0,:,:]) == 2, "gf_struct not with implemented Nambu self-consistency compatible"
        if isinstance(mu, float):
            mu = self._get_mu_matrix(mu)
        pauli3 = np.array([[1,0],[0,-1]])
        for bn, b in self.gf:
            tmp1 = pauli3.dot(self.mu[bn] - self.t_loc[bn])
            tmp2 = double_dot_product(pauli3, g_loc.gf[self.bs_map[bn]], pauli3)
            b << inverse(iOmega_n  + tmp1 - self.t**2 * tmp2)
