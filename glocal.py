import numpy as np, itertools as itt
from pytriqs.gf.local import BlockGf, GfImFreq, delta, iOmega_n, inverse, TailGf
from pytriqs.utility.bound_and_bisect import bound_and_bisect
from pytriqs.gf.local.descriptor_base import Function

from greensfunctions import MatsubaraGreensFunction
from gfoperations import double_dot_product


class GLocal(MatsubaraGreensFunction):

    def __init__(self, block_names, block_states, beta, n_iw, t, t_loc, g_loc = None):
        MatsubaraGreensFunction.__init__(self, block_names, block_states, beta, n_iw)
        if not g_loc is None:
            self.gf = g_loc.gf.copy()
        self._gf_lastloop = BlockGf(name_list = self.block_names,
                               block_list = [GfImFreq(indices = states,
                                                      beta = self.beta,
                                                      n_points = self.n_iw) for states in self.block_states])
        self.t_loc = t_loc
        self.t = t
        self.filling_with_old_mu = None
        self.last_found_mu_number = None
        self.last_found_density = None

    def set(self, selfenergy, mu, w1, w2, filling = None, dmu_max = None, *args, **kwargs):
        if filling is None:
            assert type(mu) == float or isinstance(mu, dict), "Unexpected type or class of mu."
            if type(mu) == float:
                self.set_mu_number(mu, selfenergy, w1, w2)
            else:
                self.calculate(mu, selfenergy, w1, w2)
        else:
            mu = self.find_and_set_mu(filling, selfenergy, mu, dmu_max, w1, w2)
        return mu
        
    def calculate(self, mu, selfenergy, w1, w2):
        for sk, b in self.gf:
            orbitals = [i for i in range(b.data.shape[1])]
            for i, j in itt.product(orbitals, orbitals):
                for n, iwn in enumerate(b.mesh):
                    z = lambda iw: iw + mu[sk][i, j] - self.t_loc[sk][i, j] - selfenergy.gf[sk].data[n,i,j]
                    gf = lambda iw: (z(iw) - complex(0, 1) * np.sign(z(iw).imag) * np.sqrt(4*self.t**2 - z(iw)**2))/(2*self.t**2)
                    b.data[n,i,j] = gf(iwn)
        self._fit_tail(w1, w2)

    def _fit_tail(self, w1, w2):
        beta = self.gf.beta
        for s, b in self.gf:
            n1 = int(beta/(2*np.pi)*w1 -.5)
            n2 = int(beta/(2*np.pi)*w2 -.5)
            known_moments = TailGf(b.N1, b.N2, 1, 1)
            known_moments[1] = np.identity(1)
            b.fit_tail(known_moments, 3, n1, n2)
        
    def _average(self, blockmatrix):
        d = 0
        s = 0
        for name, block in blockmatrix.items():
            for i in range(len(block)):
                d += 1
                s += block[i, i]
        return s /float(d)

    def set_mu_number(self, mu_number, selfenergy, w1, w2):
        """Assumes same mu in all orbitals"""
        mu = dict()
        for name, matrix in self.t_loc.items():
            mu[name] = np.identity(len(matrix)) * mu_number
        self.calculate(mu, selfenergy, w1, w2)

    def set_mu_get_filling(self, mu, selfenergy, w1, w2):
        self.set_mu_number(mu, selfenergy, w1, w2)
        return self.total_density()

    def find_and_set_mu(self, filling, selfenergy, mu0, dmu_max, w1, w2, *args, **kwargs):
        """Assumes a diagonal-mu basis"""
        if not filling is None:
            self.filling_with_old_mu = self.total_density()
            f = lambda mu: self.set_mu_get_filling(mu, selfenergy, w1, w2)
            mu0_number = self._mu_number(mu0)
            self.last_found_mu_number, self.last_found_density = bound_and_bisect(f, mu0_number, filling, x_name = "mu", y_name = "filling", maxiter = 10000, *args, **kwargs)
            new_mu_number, limit_applied = self.limit(self.last_found_mu_number, mu0_number, dmu_max)
            new_mu = self._mu_matrix(new_mu_number)
            if limit_applied:
                self.calculate(new_mu, selfenergy, w1, w2)
            return new_mu

    def calc_dyson(self, g0, se):
        self.gf << inverse(inverse(g0.gf) - se.gf)

    def limit(self, x, x0, dxlim):
        if abs(x - x0) > dxlim:
            return x0 + dxlim * np.sign(x - x0), True
        return x, False

    def _mu_number(self, mu):
        for key, val in mu.items():
            mu_number = val[0, 0]
            break
        return mu_number

    def _mu_matrix(self, mu_number):
        mu = dict()
        for bname, bstates in zip(self.block_names, self.block_states):
            mu[bname] = np.identity(len(bstates)) * mu_number
        return mu


class GLocalNambu(GLocal):

    def __init__(self, block_names, block_states, beta, n_iw, t, t_loc, g0_reference, g_loc = None):
        GLocal.__init__(self, block_names, block_states, beta, n_iw, t, t_loc, g_loc = None)
        self.g0_reference = g0_reference
        
    def set(self, selfenergy, mu, w1, w2, filling = None, dmu_max = None, *args, **kwargs):
        self.gf << inverse(inverse(self.g0_reference.gf) - selfenergy.gf)
        return mu
