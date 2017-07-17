import numpy as np, itertools as itt, math, scipy
from pytriqs.gf.local import inverse, iOmega_n
from pytriqs.dos import DOSFromFunction, HilbertTransform
import pytriqs.utility.mpi as mpi

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
        self.rho_lambda = lambda x: np.exp(- np.square(x) /(2 * np.square(t))) /(t *np.sqrt(2 *np.pi))
        self.rho_wmin, self.rho_wmax, self.rho_npts = rho_wmin, rho_wmax, rho_npts

    def calculate2(self, selfenergy, mu):
        se = selfenergy.get_as_BlockGf()
        for bn, b in se:
            #eps = lambda x: [xi**2 * inverse(iOmega_n + mu[bn] - se[self.flip_spin(bn)] for xi in x]
            self[bn] << self.rho_to_g(se[bn], mu[bn], epsilon_hat = eps, test_convergence = True)
            
    def calculate(self, selfenergy, mu):
        """
        Fragments taken from TRIQS Hilbert Transf.
        """
        #init
        r = (self.rho_wmax - self.rho_wmin)/float(self.rho_npts - 1)
        eps = np.array([self.rho_wmin + r * i for i in range(self.rho_npts)])
        rho = np.array([self.rho_lambda(e) for e in eps])
        #normalize
        rho[0] *= (eps[1] - eps[0])
        rho[-1] *= (eps[-1] - eps[-2])
        for i in xrange(1, eps.shape[0] - 1):
            rho[i] *=  (eps[i+1] - eps[i])/2+(eps[i] - eps[i-1])/2
        rho /= np.sum(rho)
        #calc
        for bn, b in self:
            eps2 = np.array([x**2 * np.identity(b.N1) for x in eps])
            b.zero()
            ceta_a = b.copy()
            ceta_b = b.copy()
            ceta_ab = b.copy()
            ceta_a << iOmega_n + mu[bn] - selfenergy[bn]
            ceta_b << iOmega_n + mu[bn] - selfenergy[self.flip_spin(bn)]
            ceta_ab << ceta_a * ceta_b
            del ceta_a
            for d, e2, in itt.izip (*[mpi.slice_array(A) for A in [rho, eps2]]):
                b += ceta_b * d * inverse(ceta_ab - e2)
            b << mpi.all_reduce(mpi.world, b, lambda x, y: x+y)
            mpi.barrier()

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
