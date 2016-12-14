import numpy as np
from scipy.optimize import minimize_scalar
from pytriqs.gf.local import inverse
from pytriqs.utility.bound_and_bisect import bound_and_bisect
from pytriqs.utility.dichotomy import dichotomy

from ..greensfunctions import MatsubaraGreensFunction


class GLocalGeneric(MatsubaraGreensFunction):
    """
    parent class for GLocal for different schemes, needs __init__(...) and 
    calculate(self, selfenergy, mu, w1, w2, filling = None, dmu_max = None){...fit_tail2()} !
    where mu is a blockmatrix of structure gf_struct
    """
    def __init__(self, *args, **kwargs):
        MatsubaraGreensFunction.__init__(self, *args, **kwargs)
        self.filling_with_old_mu = None
        self.last_found_mu_number = None
        self.last_found_density = None

    def calc_dyson(self, weissfield, selfenergy):
        self << inverse(inverse(weissfield) - selfenergy)

    def set(self, selfenergy, mu, w1, w2, n_mom, filling = None, dmu_max = None, *args, **kwargs):
        """
        sets GLocal using calculate(self, mu, selfenergy, w1, w2, n_mom), uses either filling or mu
        mu can be either of blockmatrix-type or scalar
        """
        if filling is None:
            assert type(mu) in [float, int, complex] or isinstance(mu, dict), "Unexpected type or class of mu."
            self.calculate(selfenergy, mu, w1, w2, n_mom)
        else:
            mu = self.find_and_set_mu(filling, selfenergy, mu, dmu_max, w1, w2, n_mom)
        return mu

    def find_and_set_mu(self, filling, selfenergy, mu0, dmu_max, w1, w2, n_mom, *args, **kwargs):
        """
        Assumes a diagonal-mu basis
        """
        # TODO place mu in center of gap
        if not filling is None:
            self.filling_with_old_mu = self.total_density()
            f = lambda mu: self._set_mu_get_filling(selfenergy, mu, w1, w2, n_mom)
            f = FunctionWithMemory(f)
            if isinstance(mu0, dict):
                mu0 = self.mu_number(mu0)
            self.last_found_mu_number, self.last_found_density = bound_and_bisect(f, mu0, filling, dx = 1, x_name = "mu", y_name = "filling", maxiter = 1000, verbosity = self.verbosity, *args, **kwargs)
            new_mu_number, limit_applied = self.limit(self.last_found_mu_number, mu0, dmu_max)
            new_mu = self.mu_matrix(new_mu_number)
            if limit_applied:
                self.calculate(selfenergy, new_mu, w1, w2, n_mom)
            return new_mu

    def _set_mu_get_filling(self, selfenergy, mu, w1, w2, n_mom):
        """
        needed for find_and_set_mu
        """
        self.calculate(selfenergy, mu, w1, w2, n_mom)
        d = self.total_density()
        return d

    def limit(self, x, x0, dxlim):
        """
        returns the element in [x0-dxlim, x0+dxlim] that is closest to x and whether it is unequal
        to x
        """
        if abs(x - x0) > dxlim:
            return x0 + dxlim * np.sign(x - x0), True
        return x, False

    def mu_number(self, mu):
        """
        converts mu to number using the first entry
        """
        for key, val in mu.items():
            mu_number = val[0, 0]
            break
        return mu_number

    def mu_matrix(self, mu_number):
        """
        converts mu to blockmatrix in GLocal basis multiplying by 1
        """
        mu = dict()
        for bname, bsize in zip(self.blocknames, self.blocksizes):
            mu[bname] = np.identity(bsize) * mu_number
        return mu


class WeissFieldGeneric(MatsubaraGreensFunction):
    def calc_dyson(self, glocal, selfenergy):
        self << inverse(inverse(glocal) + selfenergy)

    def calc_selfconsistency(self, glocal, selfenergy, mu):
        self.calc_dyson(glocal, selfenergy)


class SelfEnergyGeneric(MatsubaraGreensFunction):
    def calc_dyson(self, weissfield, glocal):
        self << inverse(weissfield) - inverse(glocal) 


class FunctionWithMemory:
    """
    a lambda with memory; memory needed due to bound_and_bisect bound finding algorithm of triqs
    some values are evaluated multiple times
    """
    def __init__(self, function):
        self.f = function
        self.x = []
        self.y = []

    def __call__(self, x):
        is_evaluated = False
        for i, x_i in enumerate(self.x):
            if x_i == x:
                is_evaluated = True
                break
        if is_evaluated:
            y = self.y[i]
        else:
            y = self.f(x)
            self.x.append(x)
            self.y.append(y)
        return y
