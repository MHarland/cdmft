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
            self[bn] << self.rho_to_g(se[bn], mu[bn])

    def calculate_unstable(self, selfenergy, mu, fit_tail = True):
        im = complex(0, 1)
        sqrt_pi = np.sqrt(np.pi)
        for bn, b in self:
            orbitals = [i for i in range(b.data.shape[1])]
            for i, j in itt.product(orbitals, orbitals):
                for n, iwn in enumerate(b.mesh):
                    z = iwn + mu[bn][i, j] - selfenergy[bn].data[n, i, j]
                    s = np.sign(z.imag)
                    g = - im *s *sqrt_pi *np.exp(- np.square(z)) *scipy.special.erfc(- im *s *z)
                    if math.isnan(g.real) or math.isnan(g.imag):
                        b.data[n,i,j] = 0
                    else:
                        b.data[n,i,j] = g
        if fit_tail:
            self.fit_tail2(self.w1, self.w2, self.n_mom)
            print self['up'].tail
        assert not math.isnan(self.total_density()), 'g(iw) hilbert transf failed'


class WeissField(WeissFieldGeneric):

    def calc_selfconsistency(self, glocal, selfenergy, mu, *args, **kwargs):
        if isinstance(mu, float) or isinstance(mu, int): mu = self._to_blockmatrix(mu)
        for bn, b in self:
            bn = self.flip_spin(bn)
            b << inverse(inverse(glocal[bn]) + selfenergy[bn])

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

class SelfEnergy(SelfEnergyGeneric):
    pass
