import numpy as np
from pytriqs.gf.local import BlockGf, GfImFreq, delta, iOmega_n, inverse
from pytriqs.utility.bound_and_bisect import bound_and_bisect

from greensfunctions import MatsubaraGreensFunction
from gfoperations import double_dot_product


class GLocal(MatsubaraGreensFunction):

    def __init__(self, block_names, block_states, beta, n_iw, *args, **kwargs):
        MatsubaraGreensFunction.__init__(self, block_names, block_states, beta, n_iw)
        self._gf_lastloop = BlockGf(name_list = self.block_names,
                               block_list = [GfImFreq(indices = states,
                                                      beta = self.beta,
                                                      n_points = self.n_iw) for states in self.block_states])
        if "t_loc" in kwargs.keys():
            self.t_loc = kwargs["t_loc"]
        if "t" in kwargs.keys():
            self.t = kwargs["t"]
        self.filling_with_old_mu = None
        self.last_found_mu_number = None
        self.last_found_density = None

    def set_mu(self, mu, selfenergy):
        for s, b in self.gf:
            b << inverse(iOmega_n + mu[s] - self.t_loc[s] - selfenergy.gf[s])

    def get_mu(self, selfenergy):
        mu_mat = dict()
        for s, b in self.gf:
            const = b.copy()
            const << (-1) * inverse(b) + iOmega_n - self.t_loc[s] - selfenergy.gf[s]
            mu_mat[s] = const.data[self.n_iw,:,:].copy()
        return mu_mat

    def _trace(self, blockmatrix):
        d = 0
        s = 0
        for name, block in blockmatrix.items():
            for i in range(len(block)):
                d += 1
                s += block[i, i]
        return s /float(d)

    def get_mu_number(self, selfenergy):
        mu_mat = self.get_mu(selfenergy)
        return self._trace(mu_mat).real

    def set_mu_number(self, mu_number, selfenergy):
        """Assumes same mu in all orbitals"""
        mu = dict()
        for name, matrix in self.t_loc.items():
            mu[name] = np.identity(len(matrix)) * mu_number
        self.set_mu(mu, selfenergy)

    def total_density(self):
        return self.gf.total_density()

    def set_mu_get_filling(self, mu_number, selfenergy):
        self.set_mu_number(mu_number, selfenergy)
        return self.total_density()

    def find_and_set_mu(self, filling, selfenergy, *args, **kwargs):
        """Assumes a diagonal-mu basis"""
        if not filling is None:
            self.filling_with_old_mu = self.total_density()
            f = lambda mu: self.set_mu_get_filling(mu, selfenergy)
            self.last_found_mu_number, self.last_found_density = bound_and_bisect(f, self.get_mu_number(selfenergy), filling, x_name = "mu", y_name = "filling", maxiter = 10000, *args, **kwargs)
            return self.last_found_mu_number

    def calc_dyson(self, g0, se):
        self.gf << inverse(inverse(g0.gf) - se.gf)
