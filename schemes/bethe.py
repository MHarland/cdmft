import numpy as np, itertools as itt, math
from pytriqs.gf.local.descriptor_base import Function
from pytriqs.gf.local import inverse, iOmega_n

from generic import GLocalGeneric, SelfEnergyGeneric, WeissFieldGeneric
from ..gfoperations import double_dot_product


class GLocal(GLocalGeneric):
    """
    w1, w2, n_mom are used for calculate only, the fitting after the impurity solver is defined in
    the selfconsistency parameters
    """
    def __init__(self, t_bethe, t_local, w1 = None, w2 = None, n_mom = 3, *args, **kwargs):
        for bn, b in t_local.items():
            for i, j in itt.product(*[range(b.shape[0])]*2):
                if i != j:
                    assert b[i, j] == 0, "Bethe Greensfunction must be diagonal for the self-consistency condition of this class"
        GLocalGeneric.__init__(self, *args, **kwargs)
        self.t_loc = t_local
        self.t_b = t_bethe
        self.w1 = (2 * self.n_iw * .8 + 1) * np.pi / self.beta if w1 is None else w1
        self.w2 = (2 * self.n_iw + 1) * np.pi / self.beta if w2 is None else w2
        self.n_mom = n_mom

    def calculate(self, selfenergy, mu):
        for sk, b in self:
            orbitals = [i for i in range(b.data.shape[1])]
            for i, j in itt.product(orbitals, orbitals):
                for n, iwn in enumerate(b.mesh):
                    z = lambda iw: iw + mu[sk][i, j] - self.t_loc[sk][i, j] - selfenergy[sk].data[n,i,j]
                    gf = lambda iw: (z(iw) - complex(0, 1) * np.sign(z(iw).imag) * np.sqrt(4*self.t_b**2 - z(iw)**2))/(2.*self.t_b**2)
                    b.data[n,i,j] = gf(iwn)
        assert not math.isnan(self.total_density()), 'g(iw) undefined for mu = '+str(self.mu_number(mu))
        self.fit_tail2(self.w1, self.w2, self.n_mom)
        assert not math.isnan(self.total_density()), 'tail fit fail!'


class GLocalAFM(GLocal):

    def calculate(self, selfenergy, mu):
        self.checkforspins()
        for sk, b in self:
            sk = self.flip_spin(sk)
            orbitals = [i for i in range(b.data.shape[1])]
            for i, j in itt.product(orbitals, orbitals):
                for n, iwn in enumerate(b.mesh):
                    z = lambda iw: iw + mu[sk][i, j] - self.t_loc[sk][i, j] - selfenergy[sk].data[n,i,j]
                    gf = lambda iw: (z(iw) - complex(0, 1) * np.sign(z(iw).imag) * np.sqrt(4*self.t_b**2 - z(iw)**2))/(2.*self.t_b**2)
                    b.data[n,i,j] = gf(iwn)
        assert not math.isnan(self.total_density()), 'g(iw) undefined for mu = '+str(self.mu_number(mu))
        self.fit_tail2(self.w1, self.w2, self.n_mom)
        assert not math.isnan(self.total_density()), 'tail fit fail!'

    def checkforspins(self):
        for name in self.blocknames:
            assert (len(name.split("up")) == 2) ^ (len(name.split("dn")) == 2), "the strings up and dn must occur exactly once in blocknames"

    def flip_spin(self, blocklabel):
        up, dn = "up", "dn"
        if up in blocklabel:
            splittedlabel = blocklabel.split(up)
            new_label = splittedlabel[0] + dn + splittedlabel[1]
        elif dn in blocklabel:
            splittedlabel = blocklabel.split(dn)
            new_label = splittedlabel[0] + up + splittedlabel[1]
        return new_label


class GLocalWithOffdiagonals(GLocal):

    def __init__(self, t_bethe, t_local, *args, **kwargs):
        GLocalGeneric.__init__(self, *args, **kwargs)
        self.t_loc = t_local
        self.t_b = t_bethe
        self._last_g_loc_convergence = []

    def calculate(self, selfenergy, mu, n_g_loc_iterations = 1000):
        last_attempt = self.copy()
        for i in range(n_g_loc_iterations):
            for s, b in self:
                b << inverse(iOmega_n + mu[s] - self.t_loc[s] - self.t_b**2 * b - selfenergy[s])
            if self._is_converged(last_attempt):
                break
            else:
                last_attempt << self
        del last_attempt

    def _is_converged(self, g_to_compare, atol = 10e-3, rtol = 1e-15):
        conv = False
        n = self.total_density()
        n_last = g_to_compare.total_density()
        self._last_g_loc_convergence.append(abs(n-n_last))
        if np.allclose(n, n_last, rtol, atol):
            conv = True
        return conv

class WeissFieldAFM(WeissFieldGeneric):

    def calc_selfconsistency(self, glocal, selfenergy, mu, *args, **kwargs):
        if isinstance(mu, float) or isinstance(mu, int): mu = self._to_blockmatrix(mu)
        for bn, b in self:
            bn = self.flip_spin(bn)
            b << inverse(iOmega_n  + mu[bn] - glocal.t_loc[bn] - glocal.t_b**2 * glocal[bn])

    def flip_spin(self, blocklabel):
        up, dn = "up", "dn"
        if up in blocklabel:
            splittedlabel = blocklabel.split(up)
            new_label = splittedlabel[0] + dn + splittedlabel[1]
        elif dn in blocklabel:
            splittedlabel = blocklabel.split(dn)
            new_label = splittedlabel[0] + up + splittedlabel[1]
        return new_label


class WeissFieldAIAO(WeissFieldGeneric):

    def __init__(self, *args, **kwargs):
        WeissFieldGeneric.__init__(self, *args, **kwargs)
        self.index_map = {}
        for i,j in itt.product(*[range(6)]*2):
            if i < 3 and j < 3:
                self.index_map[(i,j)] = (i+3,j+3,1)
            elif i >= 3 and j >= 3:
                self.index_map[(i,j)] = (i-3,j-3,1)
            elif i < 3 and j >= 3:
                self.index_map[(i,j)] = (i+3,j-3,-1)
            elif i >= 3 and j < 3:
                self.index_map[(i,j)] = (i-3,j+3,-1)

    def calc_selfconsistency(self, glocal, selfenergy, mu, *args, **kwargs):
        """
        maps a 180 deg spin rotation
        """
        tmp = self.copy()
        if isinstance(mu, float) or isinstance(mu, int): mu = self._to_blockmatrix(mu)
        for bn, b in tmp:
            for lind, rindsign in self.index_map.items():
                sign = rindsign[2]
                rind = (rindsign[0], rindsign[1])
                dij = int(lind[0] == lind[1])
                b[lind] <<  dij * iOmega_n + mu[bn][rind] - glocal.t_loc[bn][rind] - sign * glocal.t_b**2 * glocal[bn][rind]
        self << inverse(tmp)


class SelfEnergy(SelfEnergyGeneric):
    pass


class WeissField(WeissFieldGeneric):
    def calc_selfconsistency(self, glocal, selfenergy, mu, *args, **kwargs):
        if isinstance(mu, float) or isinstance(mu, int): mu = self._to_blockmatrix(mu)
        for bn, b in self:
            b << inverse(iOmega_n  + mu[bn] - glocal.t_loc[bn] - glocal.t_b**2 * glocal[bn])


class GLocalNambu(GLocal):

    def __init__(self, block_names, block_states, beta, n_iw, t, t_loc, g0_reference, g_loc = None):
        GLocal.__init__(self, block_names, block_states, beta, n_iw, t, t_loc, g_loc = None)
        self.g0_reference = g0_reference
        
    def set(self, selfenergy, mu, w1, w2, filling = None, dmu_max = None, *args, **kwargs):
        self << inverse(inverse(self.g0_reference) - selfenergy)
        return mu
