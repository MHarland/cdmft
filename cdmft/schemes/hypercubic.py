import numpy as np, itertools as itt, math, scipy
from pytriqs.gf.local import inverse, iOmega_n, TailGf
from pytriqs.dos import DOSFromFunction, HilbertTransform
import pytriqs.utility.mpi as mpi

from common import GLocalCommon, WeissFieldCommon, SelfEnergyCommon


#scipy.special:
#wofz = exp(-z**2) * erfc(-i*z)
#erfc = 1 - erf(x)
#erf = 2/sqrt(pi)*integral(exp(-t**2), t=0..z)

class GLocal(GLocalCommon):
    """
    w1, w2, n_mom are used for calculate only, the fitting after the impurity solver is defined in
    the selfconsistency parameters
    """
    def __init__(self, t, rho_wmin, rho_wmax, rho_npts, w1, w2, n_mom, *args, **kwargs):
        GLocalCommon.__init__(self, *args, **kwargs)
        self.w1 = (2 * self.n_iw * .8 + 1) * np.pi / self.beta if w1 is None else w1
        self.w2 = (2 * self.n_iw + 1) * np.pi / self.beta if w2 is None else w2
        self.n_mom = n_mom
        self.rho_lambda = lambda x: np.exp(- np.square(x) /(2 * np.square(t))) /(t *np.sqrt(2 *np.pi))
        self.rho_wmin, self.rho_wmax, self.rho_npts = rho_wmin, rho_wmax, rho_npts

    def calculate(self, selfenergy, mu):
        """
        Fragments taken from TRIQS Hilbert Transf.
        TODO solve the tail(-2) problem
        """
        #init
        selfenergy = selfenergy.get_as_BlockGf()
        g = self.get_as_BlockGf()
        gceta_a = g.copy()
        inv_gceta_b = g.copy()
        #gceta_ab = g.copy()
        gtmp = g.copy()
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
        for bn, b in g:
            eps2 = np.array([x * x * np.identity(b.N1) for x in eps])
            rhomats = np.array([r * np.identity(b.N1) for r in rho])
            mumat = np.identity(1) * mu[bn]
            b.zero()
            tmp = gtmp[bn]
            gceta_a[bn] << iOmega_n + mumat - selfenergy[bn]
            inv_gceta_b[bn] << iOmega_n + mumat - selfenergy[self.flip_spin(bn)]
            inv_gceta_b[bn].invert()
            #gceta_ab[bn] << gceta_a[bn] * gceta_b[bn]
            #print gceta_ab[bn].tail
            for d, e2 in itt.izip (*[mpi.slice_array(A) for A in [rho, eps2]]):
                tmp << gceta_a[bn] - float(e2) * inv_gceta_b[bn]
                tmp.invert()
                b += d * tmp
            b << mpi.all_reduce(mpi.world, b, lambda x, y: x+y)
            mpi.barrier()
            self[bn] << b

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


class WeissField(WeissFieldCommon):
    pass


class SelfEnergy(SelfEnergyCommon):
    pass
