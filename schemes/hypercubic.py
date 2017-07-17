import numpy as np, itertools as itt, math, scipy
from pytriqs.gf.local import inverse, iOmega_n
from pytriqs.dos import DOSFromFunction, HilbertTransform

from generic import GLocalGeneric, WeissFieldGeneric, SelfEnergyGeneric


#scipy.special:
#wofz = exp(-z**2) * erfc(-i*z)
#erfc = 1 - erf(x)
#erf = 2/sqrt(pi)*integral(exp(-t**2), t=0..z)

class GLocal(GLocalGeneric):
    """
    w1, w2, n_mom are used for calculate only, the fitting after the impurity solver is defined in
    the selfconsistency parameters
    """
    def __init__(self, t, rho_wmin, rho_wmax, rho_npts, w1, w2, n_mom, *args, **kwargs):
        GLocalGeneric.__init__(self, *args, **kwargs)
        self.w1 = (2 * self.n_iw * .8 + 1) * np.pi / self.beta if w1 is None else w1
        self.w2 = (2 * self.n_iw + 1) * np.pi / self.beta if w2 is None else w2
        self.n_mom = n_mom
        rho_lambda = lambda x: np.exp(- np.square(x) /(2 * np.square(t))) /(t *np.sqrt(2 *np.pi))
        rho = DOSFromFunction(rho_lambda, rho_wmin, rho_wmax, rho_npts)
        self.rho_to_g = HilbertTransform(rho)

    def calculate(self, selfenergy, mu):
        se = selfenergy.get_as_BlockGf()
        for bn, b in se:
            self[bn] << self.rho_to_g(se[self.flip_spin(bn)], mu[bn])

    def flip_spin(self, blocklabel):
        up, dn = "up", "dn"
        if up in blocklabel:
            splittedlabel = blocklabel.split(up)
            new_label = splittedlabel[0] + dn + splittedlabel[1]
        elif dn in blocklabel:
            splittedlabel = blocklabel.split(dn)
            new_label = splittedlabel[0] + up + splittedlabel[1]
        else:
            assert False, "spin flip failed, spin-labels not recognized"
        return new_label


class WeissField(WeissFieldGeneric):
    pass


class SelfEnergy(SelfEnergyGeneric):
    pass
