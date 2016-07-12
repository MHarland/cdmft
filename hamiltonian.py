import numpy as np
from pytriqs.operators import c as C, c_dag as CDag, n as N, dagger

from gfoperations import sum
from transformation import GfStructTransformationIndex


class HubbardSite:
    
    def __init__(self, u, spins):
        self.up = spins[0]
        self.dn = spins[1]
        self.h_int = u * N(self.up, 0) * N(self.dn, 0)

    def get_h_int(self):
        return self.h_int

    def get_gf_struct(self):
        return [[self.up, range(1)], [self.dn, range(1)]]


class HubbardPlaquette:

    def __init__(self, u, spins):
        self.up = spins[0]
        self.dn = spins[1]
        self.h_int = u * sum([N(self.up, i) * N(self.dn, i) for i in range(4)])

    def get_h_int(self):
        return self.h_int

    def get_gf_struct(self):
        return [[self.up, range(4)], [self.dn, range(4)]]


class HubbardPlaquetteMomentum:

    def __init__(self, u, spins, momenta, transformation):
        up = spins[0]
        dn = spins[1]
        sites = range(4)
        self.transformation = transformation
        self.block_labels = [spin+"-"+k for spin in spins for k in momenta]
        self.gf_struct = [[l, range(1)] for l in self.block_labels]
        self._to_mom = GfStructTransformationIndex(self.gf_struct, [[up, sites], [dn, sites]])
        self.h_int = np.sum([u * self._c_dag(up, i) * self._c(up, i) * self._c_dag(dn, i) * self._c(dn, i) for i in sites], axis = 0)

    def _c(self, spin, site):
        return sum([self.transformation[spin][site, k_index] * C(*self._to_mom(spin, k_index)) for k_index in range(4)])

    def _c_dag(self, spin, site):
        return dagger(self._c(spin, site))

    def get_h_int(self):
        return self.h_int

    def get_gf_struct(self):
        return self.gf_struct


class HubbardPlaquetteMomentumNambu:

    def __init__(self, u, spins, momenta, transformation):
        up = spins[0]
        dn = spins[1]
        self.up = up
        self.dn = dn
        sites = range(4)
        self.site_to_mom = dict([(i, momenta[i]) for i in sites])
        self.transformation = transformation
        self.block_labels = [k for k in momenta]
        self.gf_struct = [[l, range(2)] for l in self.block_labels]
        self.h_int = np.sum([u * self._c_dag(up, i) * self._c(up, i) * self._c_dag(dn, i) * self._c(dn, i) for i in sites], axis = 0)

    def _c(self, spin, site):
        if spin == self.up:
            return sum([self.transformation[spin][site, k_index] * C(self.site_to_mom[k_index], 0) for k_index in range(4)])
        elif spin == self.dn:
            return sum([self.transformation[spin][site, k_index] * CDag(self.site_to_mom[k_index], 1) for k_index in range(4)])
        assert False, "spin "+spin+" not recognized"

    def _c_dag(self, spin, site):
        return dagger(self._c(spin, site))

    def get_h_int(self):
        return self.h_int

    def get_gf_struct(self):
        return self.gf_struct
