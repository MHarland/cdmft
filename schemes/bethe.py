import numpy as np, itertools as itt, math
from pytriqs.gf.local.descriptor_base import Function
from pytriqs.gf.local import inverse, iOmega_n
from pytriqs.utility.bound_and_bisect import bound_and_bisect
from pytriqs.utility import mpi

from generic import GLocalGeneric, SelfEnergyGeneric, WeissFieldGeneric, FunctionWithMemory
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


class GLocalWithOffdiagonals(GLocalGeneric):

    def __init__(self, t_bethe, t_local, *args, **kwargs):
        GLocalGeneric.__init__(self, *args, **kwargs)
        self.t_loc = t_local
        self.t_b = t_bethe
        self._last_g_loc_convergence = []
        self._g_flipped = self.copy()#self.get_as_BlockGf().copy()
        self._last_attempt = self.copy()#self.get_as_BlockGf().copy()

    def _set_g_flipped(self):
        for s, b in self:
            self._g_flipped[self.flip_spin(s)] << b

    def calculate(self, selfenergy, mu, n_g_loc_iterations = 1000):
        self._set_g_flipped()
        for i in range(n_g_loc_iterations):
            self.calc_selfconsistency(selfenergy, mu)
            if self._is_converged(self._last_attempt):
                break
            else:
                self._last_attempt << self
        if mpi.is_master_node():
            print 'GLocal convergence took '+str(i)+' iterations'

    def calc_selfconsistency(self, selfenergy, mu):
        for s, b in self:
            b << inverse(iOmega_n + mu[s] - self.t_loc[s] - self.t_b**2 * self._g_flipped[s] - selfenergy[s])

    def _is_converged(self, g_to_compare, atol = 10e-3, rtol = 1e-15):
        conv = False
        n = self.total_density()
        n_last = g_to_compare.total_density()
        self._last_g_loc_convergence.append(abs(n-n_last))
        if np.allclose(n, n_last, rtol, atol):
            # TODO compare G
            conv = True
        return conv


class GLocalInhomogeneous(GLocalWithOffdiagonals):

    def calc_selfconsistency(self, selfenergy, mu):
        for s, b in self:
            b << inverse(iOmega_n + mu[s] - self.t_loc[s] - double_dot_product(self.t_b[s], self._g_flipped[s], self.t_b[s]) - selfenergy[s])


class GLocalAIAO(GLocalWithOffdiagonals):

    def __init__(self, *args, **kwargs):
        GLocalWithOffdiagonals.__init__(self, *args, **kwargs)
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
    
    def _set_g_flipped(self):
        for s, b in self:
            for lind, rindsign in self.index_map.items():
                sign = rindsign[2]
                rind = (rindsign[0], rindsign[1])
                dij = int(lind[0] == lind[1])
                self._g_flipped[s][lind] << sign * b[rind]


class WeissField(WeissFieldGeneric):
    
    def calc_selfconsistency(self, glocal, selfenergy, mu, *args, **kwargs):
        if isinstance(mu, float) or isinstance(mu, int): mu = self._to_blockmatrix(mu)
        for bn, b in self:
            b << inverse(iOmega_n  + mu[bn] - glocal.t_loc[bn] - glocal.t_b**2 * glocal[bn])


class WeissFieldAFM(WeissFieldGeneric):
    
    def calc_selfconsistency(self, glocal, selfenergy, mu, *args, **kwargs):
        if isinstance(mu, float) or isinstance(mu, int): mu = self._to_blockmatrix(mu)
        for bn, b in self:
            bn = self.flip_spin(bn)
            b << inverse(iOmega_n  + mu[bn] - glocal.t_loc[bn] - glocal.t_b**2 * glocal[bn])


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


class WeissFieldInhomogeneous(WeissFieldGeneric):

    def calc_selfconsistency(self, glocal, selfenergy, mu, *args, **kwargs):
        if isinstance(mu, float) or isinstance(mu, int): mu = self._to_blockmatrix(mu)
        for bn, b in self:
            bn = self.flip_spin(bn)
            b << inverse(iOmega_n  + mu[bn] - glocal.t_loc[bn] - double_dot_product(glocal.t_b[bn], glocal[bn], glocal.t_b[bn]))


class WeissFieldNambu(WeissFieldGeneric):
    """
    with afm and allows for imaginary gap, too
    """
    def __init__(self, *args, **kwargs):
        WeissFieldGeneric.__init__(self, *args, **kwargs)
        self._tmp = self.copy()
        self._ceta = self.copy()
    
    def calc_selfconsistency(self, glocal, selfenergy, mu, *args, **kwargs):
        if isinstance(mu, float) or isinstance(mu, int): mu = self._to_blockmatrix(mu)
        pauli3 = np.array([[1, 0], [0, -1]])
        for bn, b in self:
            ceta = self._ceta[bn]
            ceta << iOmega_n  + (mu[bn] - glocal.t_loc[bn]).dot(pauli3)
            self._tmp[bn][0, 0] << ceta[0, 0] - glocal.t_b**2 * glocal[bn][1, 1]
            self._tmp[bn][1, 1] << ceta[1, 1] - glocal.t_b**2 * glocal[bn][0, 0]
            self._tmp[bn][0, 1] << ceta[0, 1] - glocal.t_b**2 * (-1) * glocal[bn][0, 1]
            self._tmp[bn][1, 0] << ceta[1, 0] - glocal.t_b**2 * (-1) * glocal[bn][1, 0]
        self << inverse(self._tmp)


class WeissFieldAFMNambu(WeissFieldNambu): # TODO
    """
    with afm and allows for imaginary gap, too
    """
    def calc_selfconsistency(self, glocal, selfenergy, mu, *args, **kwargs):
        if isinstance(mu, float) or isinstance(mu, int): mu = self._to_blockmatrix(mu)
        pauli3 = np.array([[1, 0], [0, -1]])
        one = np.identity(2)
        for bn, b in self:
            ceta = self._ceta[bn]
            ceta << iOmega_n  + (mu[bn] - glocal.t_loc[bn]).dot(np.kronecker(one, pauli3))
            #momentum-diagonals:
            self._tmp[bn][0, 0] << ceta[0, 0] - glocal.t_b**2 * (-1) * glocal[bn][1, 1].conjugate()
            self._tmp[bn][1, 1] << ceta[1, 1] - glocal.t_b**2 * (-1) * glocal[bn][0, 0].conjugate()
            self._tmp[bn][2, 2] << ceta[2, 2] - glocal.t_b**2 * (-1) * glocal[bn][3, 3].conjugate()
            self._tmp[bn][3, 3] << ceta[3, 3] - glocal.t_b**2 * (-1) * glocal[bn][2, 2].conjugate()
            #momentum-off-diagonals:
            self._tmp[bn][0, 2] << ceta[0, 2] - glocal.t_b**2 * (-1) * glocal[bn][1, 3].conjugate()
            self._tmp[bn][1, 3] << ceta[1, 3] - glocal.t_b**2 * (-1) * glocal[bn][0, 2].conjugate()
            self._tmp[bn][2, 0] << ceta[2, 0] - glocal.t_b**2 * (-1) * glocal[bn][3, 1].conjugate()
            self._tmp[bn][3, 1] << ceta[3, 1] - glocal.t_b**2 * (-1) * glocal[bn][2, 0].conjugate()
            #nambu-off-diagonals:
            indices = [(0,1), (0,3), (1,0), (1,2), (2,1), (2,3), (3,0), (3,2)]
            for i in indices:
                self._tmp[bn][i] << ceta[i] - glocal.t_b**2 * (-1) * glocal[bn][i]
        self << inverse(self._tmp)


class GLocalNambu(GLocalWithOffdiagonals):

    def __init__(self, *args, **kwargs):
        GLocalWithOffdiagonals.__init__(self, *args, **kwargs)
        self.p3 = np.array([[1, 0], [0, -1]])

    def _set_g_flipped(self):
        for s, b in self:
            self._g_flipped[s][0, 0] << b[1, 1]
            self._g_flipped[s][1, 1] << b[0, 0]
            self._g_flipped[s][0, 1] << b[0, 1]
            self._g_flipped[s][1, 0] << b[1, 0]

    def calc_selfconsistency(self, selfenergy, mu):
        for s, b in self:
            b << inverse(iOmega_n + (mu[s] - self.t_loc[s]).dot(self.p3) - self.t_b**2 * double_dot_product(self.p3, self._g_flipped[s], self.p3) - selfenergy[s])

    def total_density_nambu(self, g = None):
        if g is None: g = self
        densities = []
        for s, b in g:
            densities.append(b[0, 0].total_density())
            densities.append(- b[1, 1].conjugate().total_density())
        density = np.sum(densities)
        return density

    def _is_converged(self, g_to_compare, atol = 10e-3, rtol = 0, g_atol = 10e-3, n_freq_to_compare = 20):
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
                x = self[b][i, j].data[self.iw_offset:self.iw_offset+ n_freq_to_compare, 0, 0]
                y = g_to_compare[b][i, j].data[self.iw_offset:self.iw_offset+ n_freq_to_compare, 0, 0]
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
            self.filling_with_old_mu = self.total_density_nambu()
            f = lambda mu: self._set_mu_get_filling(selfenergy, mu)
            f = FunctionWithMemory(f)
            self.last_found_mu_number, self.last_found_density = bound_and_bisect(f, mu0, filling, dx = self.mu_dx, x_name = "mu", y_name = "filling", maxiter = self.mu_maxiter, verbosity = self.verbosity)
            new_mu, limit_applied = self.limit(self.last_found_mu_number, mu0, dmu_max)
            if limit_applied:
                self.calculate(selfenergy, self.make_matrix(new_mu))
            return new_mu

    def _set_mu_get_filling(self, selfenergy, mu):
        """
        needed for find_and_set_mu
        """
        self.calculate(selfenergy, self.make_matrix(mu))
        d = self.total_density_nambu()
        print d.real
        return d


class GLocalAFMNambu(GLocalNambu):
    """
    GLocalNambu with broken cluster symmetry(afm)
    """
    def _set_g_flipped(self):
        for s, b in self:
            self._g_flipped[s][0, 0] << b[1, 1]
            self._g_flipped[s][1, 1] << b[0, 0]
            self._g_flipped[s][0, 1] << b[1, 0]
            self._g_flipped[s][1, 0] << b[0, 1]

    def calc_selfconsistency(self, selfenergy, mu): # TODO
        for s, b in self:
            b << inverse(iOmega_n + (mu[s] - self.t_loc[s]).dot(self.p3) - self.t_b**2 * double_dot_product(self.p3, self._g_flipped[s], self.p3) - selfenergy[s])

    def total_density_nambu(self, g = None): # TODO
        if g is None: g = self
        densities = []
        for s, b in g:
            densities.append(b[0, 0].total_density())
            densities.append(- b[1, 1].conjugate().total_density())
        density = np.sum(densities)
        return density


class SelfEnergy(SelfEnergyGeneric):
    pass
