import numpy as np
import itertools as itt
from pytriqs.gf import inverse, iOmega_n
from pytriqs.lattice.tight_binding import TBLattice
from pytriqs.utility import mpi
from pytriqs.utility.bound_and_bisect import bound_and_bisect
from pytriqs.sumk import SumkDiscreteFromLattice

from common import GLocalCommon, SelfEnergyCommon, WeissFieldCommon, FunctionWithMemory
from ..gfoperations import double_dot_product


class GLocal(GLocalCommon):
    """
    RevModPhys.77.1027
    transf_for_ksum allows for a unitary transformation for the k-summation of G_local; it is automatically 
    backtransformed after the summation
    """

    def __init__(self, lattice_dispersion, transf_for_ksum, *args, **kwargs):
        self.lat = lattice_dispersion
        self.transfksum = transf_for_ksum
        assert hasattr(self.lat, 'bz_points') and hasattr(self.lat, 'bz_weights') and hasattr(
            self.lat, 'energies'), 'make sure lattice_dispersion has the attributes bz_points, bz_weights and energies!'
        self.bz = [self.lat.bz_points, self.lat.bz_weights, self.lat.energies]
        GLocalCommon.__init__(self, *args, **kwargs)

    def calculate(self, selfenergy, mu):
        g = self.get_as_BlockGf()  # TODO need BlockGf for __iadd__ and reduce
        if self.transfksum is not None:
            g = self.transfksum.transform(g)
            selfenergy = self.transfksum.transform(selfenergy)
            mu = self.transfksum.transform(mu)
        g.zero()
        result = g.copy()
        for k, w, d in itt.izip(*[mpi.slice_array(x) for x in self.bz]):  # TODO c++
            for bn, b in g:
                b << b + w * inverse(iOmega_n +
                                     mu[bn] - d[bn] - selfenergy[bn])
        result << mpi.all_reduce(mpi.world, g, lambda x, y: x + y)
        if self.transfksum is not None:
            result = self.transfksum.backtransform(result)
        self << result
        mpi.barrier()


class SelfEnergy(SelfEnergyCommon):
    pass


class WeissField(WeissFieldCommon):
    pass


class GLocalNambu(GLocal):
    """
    """

    def calculate(self, selfenergy, mu):
        p3 = np.kron(np.array([[1, 0], [0, -1]]), np.eye(4))
        g = self.get_as_BlockGf()  # TODO need BlockGf for __iadd__ and reduce
        if self.transfksum is not None:
            g = self.transfksum.transform(g)
            selfenergy = self.transfksum.transform(selfenergy)
            mu = self.transfksum.transform(mu)
        g.zero()
        result = g.copy()
        for k, w, d in itt.izip(*[mpi.slice_array(x) for x in self.bz]):  # TODO c++
            for bn, b in g:
                b << b + w * inverse(iOmega_n +
                                     (mu[bn] - d[bn]).dot(p3) - selfenergy[bn])
        result << mpi.all_reduce(mpi.world, g, lambda x, y: x + y)
        if self.transfksum is not None:
            result = self.transfksum.backtransform(result)
        self << result
        mpi.barrier()

    def total_density_nambu(self, g=None):
        if g is None:
            g = self
        densities = []
        for s, b in g:
            densities.append(b[0, 0].density().real)
            densities.append(- b[1, 1].conjugate().density().real)
        density = np.sum(densities)
        return density

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


class WeissFieldNambu(WeissFieldCommon):
    """
    allows for plaquette-afm
    """

    def calc_selfconsistency(self, glocal, selfenergy, mu):
        tmp = self.get_as_BlockGf()
        for bn, b in self:
            b << inverse(inverse(glocal[bn]) + selfenergy[bn])
            tmp[bn] = b.copy()
            b[0, 0] << -1 * tmp[bn][1, 1].conjugate()
            b[1, 1] << -1 * tmp[bn][0, 0].conjugate()
