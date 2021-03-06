import numpy as np
import itertools as itt
import math
from pytriqs.gf.descriptor_base import Function
from pytriqs.gf import inverse, iOmega_n
from pytriqs.utility.bound_and_bisect import bound_and_bisect
from pytriqs.utility import mpi

from common import GLocalCommon, SelfEnergyCommon, WeissFieldCommon, FunctionWithMemory
from ..gfoperations import double_dot_product


class GLocal(GLocalCommon):
    """
    w1, w2, n_mom are used for calculate only, the fitting after the impurity solver is defined in
    the selfconsistency parameters
    """

    def __init__(self, t_bethe, t_local, w1=None, w2=None, n_mom=3, *args, **kwargs):
        for bn, b in t_local.items():
            for i, j in itt.product(*[range(b.shape[0])]*2):
                if i != j:
                    assert b[i, j] == 0, "Bethe Greensfunction must be diagonal for the self-consistency condition of this class"
        GLocalCommon.__init__(self, *args, **kwargs)
        self.t_loc = t_local
        self.t_b = t_bethe
        self.w1 = (2 * self.n_iw * .8 + 1) * np.pi / \
            self.mesh.beta if w1 is None else w1
        self.w2 = (2 * self.n_iw + 1) * np.pi / \
            self.mesh.beta if w2 is None else w2
        self.n_mom = n_mom

    def calculate(self, selfenergy, mu):
        for sk, b in self:
            orbitals = [i for i in range(b.data.shape[1])]
            for i, j in itt.product(orbitals, orbitals):
                for n, iwn in enumerate(b.mesh):
                    def z(iw): return iw + mu[sk][i, j] - \
                        self.t_loc[sk][i, j] - selfenergy[sk].data[n, i, j]
                    def gf(iw): return (z(iw) - complex(0, 1) * np.sign(z(iw).imag)
                                        * np.sqrt(4*self.t_b**2 - z(iw)**2))/(2.*self.t_b**2)
                    b.data[n, i, j] = gf(iwn)

        assert not math.isnan(self.total_density(
        ).real), 'g(iw) undefined for mu = '+str(self.mu_number(mu))
        self.fit_tail2(fit_min_w=self.w1, fit_max_w=self.w2,
                       fit_max_moment=self.n_mom)
        assert not math.isnan(self.total_density().real), 'tail fit fail!'


class GLocalAFM(GLocal):

    def calculate(self, selfenergy, mu):
        for sk, b in self:
            sk = self.flip_spin(sk)
            orbitals = [i for i in range(b.data.shape[1])]
            for i, j in itt.product(orbitals, orbitals):
                for n, iwn in enumerate(b.mesh):
                    def z(iw): return iw + mu[sk][i, j] - \
                        self.t_loc[sk][i, j] - selfenergy[sk].data[n, i, j]
                    def gf(iw): return (z(iw) - complex(0, 1) * np.sign(z(iw).imag)
                                        * np.sqrt(4*self.t_b**2 - z(iw)**2))/(2.*self.t_b**2)
                    b.data[n, i, j] = gf(iwn)
        assert not math.isnan(self.total_density(
        ).real), 'g(iw) undefined for mu = '+str(self.mu_number(mu))
        self.fit_tail2()
        assert not math.isnan(self.total_density().real), 'tail fit fail!'


class GLocalWithOffdiagonals(GLocalCommon):

    def __init__(self, t_bethe, t_local, *args, **kwargs):
        GLocalCommon.__init__(self, *args, **kwargs)
        self.t_loc = t_local
        self.t_b = t_bethe
        self._last_g_loc_convergence = []
        self._g_flipped = self.copy()
        self._last_attempt = self.copy()

    def _set_g_flipped(self):
        for s, b in self:
            self._g_flipped[self.flip_spin(s)] << b

    def calculate(self, selfenergy, mu, n_g_loc_iterations=1000):
        self._set_g_flipped()
        for i in range(n_g_loc_iterations):
            self.calc_selfconsistency(selfenergy, mu)
            if self._is_converged(self._last_attempt):
                break
            else:
                self._last_attempt << self
        if mpi.is_master_node() and self.verbosity:
            print 'gloc convergence took', i, 'iterations'

    def calc_selfconsistency(self, selfenergy, mu):
        for s, b in self:
            b << inverse(iOmega_n + mu[s] - self.t_loc[s] -
                         self.t_b**2 * self._g_flipped[s] - selfenergy[s])

    def _is_converged(self, g_to_compare, atol=10e-3, rtol=1e-15, g_atol=10e-4, n_freq_to_compare=50):
        conv = False
        n = self.total_density()
        n_last = self.total_density()
        self._last_g_loc_convergence.append(abs(n-n_last))
        if np.allclose(n, n_last, rtol, atol):
            conv = True
            for b, i, j in self.all_indices:
                x = self[b].data[:, :, :]
                xdat = x[self.iw_offset:self.iw_offset +
                         n_freq_to_compare, i, j]
                y = g_to_compare[b].data[:, :, :]
                ydat = y[self.iw_offset:self.iw_offset +
                         n_freq_to_compare, i, j]
                if not np.allclose(xdat, ydat, rtol, g_atol):
                    conv = False
                    break
        return conv


class GLocalInhomogeneous(GLocalWithOffdiagonals):

    def calc_selfconsistency(self, selfenergy, mu):
        for s, b in self:
            b << inverse(iOmega_n + mu[s] - self.t_loc[s] - double_dot_product(
                self.t_b[s], self._g_flipped[s], self.t_b[s]) - selfenergy[s])


class GLocalInhomogeneousFM(GLocalWithOffdiagonals):

    def calc_selfconsistency(self, selfenergy, mu):
        for s, b in self:
            b << inverse(iOmega_n + mu[s] - self.t_loc[s] - double_dot_product(
                self.t_b[s], b, self.t_b[s]) - selfenergy[s])


class GLocalAIAO(GLocalWithOffdiagonals):

    def __init__(self, *args, **kwargs):
        GLocalWithOffdiagonals.__init__(self, *args, **kwargs)
        self.index_map = {}
        for i, j in itt.product(*[range(6)]*2):
            if i < 3 and j < 3:
                self.index_map[(i, j)] = (i+3, j+3, 1)
            elif i >= 3 and j >= 3:
                self.index_map[(i, j)] = (i-3, j-3, 1)
            elif i < 3 and j >= 3:
                self.index_map[(i, j)] = (i+3, j-3, -1)
            elif i >= 3 and j < 3:
                self.index_map[(i, j)] = (i-3, j+3, -1)

    def _set_g_flipped(self):
        for s, b in self:
            for lind, rindsign in self.index_map.items():
                sign = rindsign[2]
                rind = (rindsign[0], rindsign[1])
                dij = int(lind[0] == lind[1])
                self._g_flipped[s][lind] << sign * b[rind]


class WeissField(WeissFieldCommon):

    def calc_selfconsistency(self, glocal, selfenergy, mu, *args, **kwargs):
        if isinstance(mu, float) or isinstance(mu, int):
            mu = self._to_blockmatrix(mu)
        for bn, b in self:
            b << inverse(
                iOmega_n + mu[bn] - glocal.t_loc[bn] - glocal.t_b**2 * glocal[bn])


class WeissFieldAFM(WeissFieldCommon):

    def calc_selfconsistency(self, glocal, selfenergy, mu, *args, **kwargs):
        if isinstance(mu, float) or isinstance(mu, int):
            mu = self._to_blockmatrix(mu)
        for bn, b in self:
            bn = self.flip_spin(bn)
            b << inverse(
                iOmega_n + mu[bn] - glocal.t_loc[bn] - glocal.t_b**2 * glocal[bn])


class WeissFieldAIAO(WeissFieldCommon):

    def __init__(self, *args, **kwargs):
        WeissFieldCommon.__init__(self, *args, **kwargs)
        self.index_map = {}
        for i, j in itt.product(*[range(6)]*2):
            if i < 3 and j < 3:
                self.index_map[(i, j)] = (i+3, j+3, 1)
            elif i >= 3 and j >= 3:
                self.index_map[(i, j)] = (i-3, j-3, 1)
            elif i < 3 and j >= 3:
                self.index_map[(i, j)] = (i+3, j-3, -1)
            elif i >= 3 and j < 3:
                self.index_map[(i, j)] = (i-3, j+3, -1)

    def calc_selfconsistency(self, glocal, selfenergy, mu, *args, **kwargs):
        """
        maps a 180 deg spin rotation
        """
        tmp = self.copy()
        if isinstance(mu, float) or isinstance(mu, int):
            mu = self._to_blockmatrix(mu)
        for bn, b in tmp:
            for lind, rindsign in self.index_map.items():
                sign = rindsign[2]
                rind = (rindsign[0], rindsign[1])
                dij = int(lind[0] == lind[1])
                b[lind] << dij * iOmega_n + mu[bn][rind] - \
                    glocal.t_loc[bn][rind] - sign * \
                    glocal.t_b**2 * glocal[bn][rind]
        self << inverse(tmp)


class WeissFieldInhomogeneous(WeissFieldCommon):

    def calc_selfconsistency(self, glocal, selfenergy, mu, *args, **kwargs):
        if isinstance(mu, float) or isinstance(mu, int):
            mu = self._to_blockmatrix(mu)
        for bn, b in self:
            bn = self.flip_spin(bn)
            b << inverse(iOmega_n + mu[bn] - glocal.t_loc[bn] -
                         double_dot_product(glocal.t_b[bn], glocal[bn], glocal.t_b[bn]))


class WeissFieldInhomogeneousFM(WeissFieldCommon):

    def calc_selfconsistency(self, glocal, selfenergy, mu, *args, **kwargs):
        if isinstance(mu, float) or isinstance(mu, int):
            mu = self._to_blockmatrix(mu)
        for bn, b in self:
            b << inverse(iOmega_n + mu[bn] - glocal.t_loc[bn] -
                         double_dot_product(glocal.t_b[bn], glocal[bn], glocal.t_b[bn]))


class WeissFieldNambu(WeissFieldCommon):
    """
    with afm and allows for imaginary gap, too
    """

    def __init__(self, *args, **kwargs):
        WeissFieldCommon.__init__(self, *args, **kwargs)
        self._tmp = self.copy()
        self._ceta = self.copy()

    def calc_selfconsistency(self, glocal, selfenergy, mu, *args, **kwargs):
        if isinstance(mu, float) or isinstance(mu, int):
            mu = self._to_blockmatrix(mu)
        pauli3 = np.array([[1, 0], [0, -1]])
        for bn, b in self:
            ceta = self._ceta[bn]
            ceta << iOmega_n + (mu[bn] - glocal.t_loc[bn]).dot(pauli3)
            self._tmp[bn][0, 0] << ceta[0, 0] - glocal.t_b**2 * \
                (-1) * glocal[bn][1, 1].conjugate()
            self._tmp[bn][1, 1] << ceta[1, 1] - glocal.t_b**2 * \
                (-1) * glocal[bn][0, 0].conjugate()
            self._tmp[bn][0, 1] << ceta[0, 1] - \
                glocal.t_b**2 * (-1) * glocal[bn][0, 1]
            self._tmp[bn][1, 0] << ceta[1, 0] - \
                glocal.t_b**2 * (-1) * glocal[bn][1, 0]
        self << inverse(self._tmp)


class WeissFieldAFMNambu(WeissFieldNambu):
    """
    with afm and allows for imaginary gap, too
    """

    def calc_selfconsistency(self, glocal, selfenergy, mu, *args, **kwargs):
        glocal._set_g_flipped()
        one = np.identity(2)
        for bn, b in self:
            mumat = mu*np.identity(len(glocal.t_loc[bn]))
            ceta = self._ceta[bn]
            ceta << iOmega_n + \
                (mumat - glocal.t_loc[bn]).dot(np.kron(one, glocal.p3))
            self._tmp[bn] << ceta - double_dot_product(
                glocal.t_b[bn], glocal._g_flipped[bn], glocal.t_b[bn])
        self << inverse(self._tmp)


class GLocalNambu(GLocalWithOffdiagonals):

    def __init__(self, *args, **kwargs):
        GLocalWithOffdiagonals.__init__(self, *args, **kwargs)
        self.p3 = np.array([[1, 0], [0, -1]])

    def _set_g_flipped(self):
        for s, b in self:
            self._g_flipped[s][0, 0] << (-1) * b[1, 1].conjugate()
            self._g_flipped[s][1, 1] << (-1) * b[0, 0].conjugate()
            self._g_flipped[s][0, 1] << b[0, 1]
            self._g_flipped[s][1, 0] << b[1, 0]

    def calc_selfconsistency(self, selfenergy, mu):
        for s, b in self:
            b << inverse(iOmega_n + (mu[s] - self.t_loc[s]).dot(self.p3) - self.t_b**2 *
                         double_dot_product(self.p3, self._g_flipped[s], self.p3) - selfenergy[s])

    def total_density_nambu(self, g=None):
        if g is None:
            g = self
        densities = []
        for s, b in g:
            densities.append(b[0, 0].density())
            densities.append(- b[1, 1].conjugate().density())
        density = np.sum(densities)
        return density

    def _is_converged(self, g_to_compare, atol=10e-3, rtol=0, g_atol=10e-3, n_freq_to_compare=50):
        """
        checks densities first, if positive: checks components of g
        """
        conv = False
        n = self.total_density_nambu()
        n_last = self.total_density_nambu(g_to_compare)
        self._last_g_loc_convergence.append(abs(n-n_last))
        if np.allclose(n, n_last, rtol, atol):
            conv = True
            for index_triple in self.all_indices:
                b, i, j = index_triple
                x = self[b][i, j].data[self.iw_offset:self.iw_offset +
                                       n_freq_to_compare]  # , 0, 0]
                y = g_to_compare[b][i, j].data[self.iw_offset:self.iw_offset +
                                               n_freq_to_compare]  # , 0, 0]
                if not np.allclose(x, y, rtol, g_atol):
                    conv = False
                    break
        return conv

    def find_and_set_mu(self, filling, selfenergy, mu0, dmu_max):
        """
        Assumes a diagonal-mu basis
        """
        # TODO place mu in center of gap
        if not filling is None:
            self.filling_with_old_mu = self.total_density_nambu().real
            def f(mu): return self._set_mu_get_filling(selfenergy, mu)
            f = FunctionWithMemory(f)
            self.last_found_mu_number, self.last_found_density = bound_and_bisect(
                f, mu0, filling, dx=self.mu_dx, x_name="mu", y_name="filling", maxiter=self.mu_maxiter, verbosity=self.verbosity)
            new_mu, limit_applied = self.limit(
                self.last_found_mu_number, mu0, dmu_max)
            if limit_applied:
                self.calculate(selfenergy, self.make_matrix(new_mu))
            return new_mu

    def _set_mu_get_filling(self, selfenergy, mu):
        """
        needed for find_and_set_mu
        """
        self.calculate(selfenergy, self.make_matrix(mu))
        d = self.total_density_nambu().real
        return d


class GLocalAFMNambu(GLocalNambu):
    """
    GLocalNambu with broken cluster symmetry(afm)
    """

    def __init__(self, *args, **kwargs):
        GLocalNambu.__init__(self, *args, **kwargs)
        self.p3 = np.array([[1, 0], [0, -1]])
        self._tmp = self.copy()
        self._ceta = self.copy()

    def _set_g_flipped(self):
        flipmap = {(0, 0): (1, 1), (0, 2): (1, 3),
                   (2, 0): (3, 1), (2, 2): (3, 3)}
        for s, b in self:
            self._g_flipped[s] << b
            for i, j in flipmap.items():
                self._g_flipped[s][i] << (-1) * b[j].conjugate()
                self._g_flipped[s][j] << (-1) * b[i].conjugate()

    def calc_selfconsistency(self, selfenergy, mu):
        one = np.identity(2)
        for bn, b in self:
            ceta = self._ceta[bn]
            ceta << iOmega_n + (mu[bn] - self.t_loc[bn]
                                ).dot(np.kron(one, self.p3)) - selfenergy[bn]
            self._tmp[bn] << ceta - \
                double_dot_product(
                    self.t_b[bn], self._g_flipped[bn], self.t_b[bn])
        self << inverse(self._tmp)

    def total_density_nambu(self, g=None):
        if g is None:
            g = self
        densities = []
        for s, b in g:
            densities.append(b[0, 0].density())
            densities.append(- b[1, 1].conjugate().density())
            densities.append(b[2, 2].density())
            densities.append(- b[3, 3].conjugate().density())
        density = np.sum(densities)
        return density


class SelfEnergyAFMNambu(SelfEnergyCommon):
    def __init__(self, *args, **kwargs):
        SelfEnergyCommon.__init__(self, *args, **kwargs)
        self._g_flipped = self.copy()

    def _set_g_flipped(self):
        flipmap = {(0, 0): (1, 1), (0, 2): (1, 3),
                   (2, 0): (3, 1), (2, 2): (3, 3)}
        for s, b in self:
            self._g_flipped[s] << b
            for i, j in flipmap.items():
                self._g_flipped[s][i] << (-1) * b[j].conjugate()
                self._g_flipped[s][j] << (-1) * b[i].conjugate()


class SelfEnergy(SelfEnergyCommon):
    pass
