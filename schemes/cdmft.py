import numpy as np, itertools as itt
from pytriqs.gf.local import inverse, iOmega_n
from pytriqs.lattice.tight_binding import TBLattice
from pytriqs.utility import mpi
from pytriqs.sumk import SumkDiscreteFromLattice

from generic import GLocalGeneric, SelfEnergyGeneric, WeissFieldGeneric, FunctionWithMemory
from bethe import WeissFieldNambu
from ..gfoperations import double_dot_product


class GLocal(GLocalGeneric):
    """
    RevModPhys.77.1027
    """
    def __init__(self, lattice_dispersion, *args, **kwargs):
        self.lat = lattice_dispersion
        self.p3 = np.array([[1, 0], [0, -1]])
        assert hasattr(self.lat, 'bz_points') and hasattr(self.lat, 'bz_weights') and hasattr(self.lat, 'energies'), 'make sure lattice_dispersion has the attributes bz_points, bz_weights and energies!'
        self.bz = [self.lat.bz_points, self.lat.bz_weights, self.lat.energies]
        GLocalGeneric.__init__(self, *args, **kwargs)

    def calculate(self, selfenergy, mu):
        g = self.get_as_BlockGf() # TODO need BlockGf for __iadd__ and reduce
        g.zero()
        for k, w, d in itt.izip(*[mpi.slice_array(x) for x in self.bz]): # TODO c++
            for bn, b in g:
                b << b + w * inverse(iOmega_n + mu[bn] - d[bn] - selfenergy[bn])
        self << mpi.all_reduce(mpi.world, g, lambda x, y: x + y)
        mpi.barrier()


class SelfEnergy(SelfEnergyGeneric):
    pass


class WeissField(WeissFieldGeneric):
    pass

class GLocalNambu(GLocal):
    """
    """
    def calculate(self, selfenergy, mu):
        g = self.get_as_BlockGf() # TODO need BlockGf for __iadd__ and reduce
        g.zero()
        for k, w, d in itt.izip(*[mpi.slice_array(x) for x in self.bz]): # TODO c++
            for bn, b in g:
                b << b + w * inverse(iOmega_n + (mu[bn] - d[bn]).dot(self.p3) - selfenergy[bn])
        self << mpi.all_reduce(mpi.world, g, lambda x, y: x + y)
        mpi.barrier()

    def total_density_nambu(self, g = None):
        if g is None: g = self
        densities = []
        for s, b in g:
            densities.append(b[0, 0].total_density())
            densities.append(- b[1, 1].conjugate().total_density())
        density = np.sum(densities)
        return density

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
        return d
