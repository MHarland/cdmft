import numpy as np, itertools as itt
from pytriqs.gf.local import inverse, iOmega_n
from pytriqs.lattice.tight_binding import TBLattice
from pytriqs.utility import mpi
from pytriqs.sumk import SumkDiscreteFromLattice

from generic import GLocalGeneric, SelfEnergyGeneric, WeissFieldGeneric
from ..gfoperations import double_dot_product


class GLocal(GLocalGeneric):
    """
    RevModPhys.77.1027
    """
    def __init__(self, lattice_dispersion, *args, **kwargs):
        self.lat = lattice_dispersion
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
