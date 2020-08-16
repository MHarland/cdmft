import numpy as np
import itertools as itt
from pytriqs.gf import inverse, iOmega_n
from pytriqs.utility import mpi

from common import GLocalCommon, SelfEnergyCommon, WeissFieldCommon


class GLocal(GLocalCommon):
    """
    RevModPhys.77.1027
    """

    def __init__(self, lattice_dispersion, *args, **kwargs):
        self.lat = lattice_dispersion
        assert hasattr(self.lat, 'bz_points') and hasattr(self.lat, 'bz_weights') and hasattr(
            self.lat, 'energies'), 'make sure lattice_dispersion has the attributes bz_points, bz_weights and energies!'
        self.bz = [self.lat.bz_points, self.lat.bz_weights, self.lat.energies]
        GLocalCommon.__init__(self, *args, **kwargs)
        spins = [s for s in self.indices]

    def calculate(self, selfenergy, mu):
        g = self.get_as_BlockGf()  # TODO need BlockGf for __iadd__ and reduce
        g.zero()
        for k, w, d in itt.izip(*[mpi.slice_array(x) for x in self.bz]):  # TODO c++
            for bn, b in g:
                g[bn] += inverse(iOmega_n + mu[bn] -
                                 d[bn] - selfenergy[bn]) * w
        self << mpi.all_reduce(mpi.world, g, lambda x, y: x + y)
        mpi.barrier()


class SelfEnergy(SelfEnergyCommon):
    pass


class WeissField(WeissFieldCommon):

    def calc_selfconsistency(self, glocal, selfenergy, mu):
        spins = [s for s in self.indices]
        flip = {spins[0]: spins[1], spins[1]: spins[0]}
        for s, b in self:
            self[s] << inverse(inverse(glocal[flip[s]]) + selfenergy[flip[s]])
