import numpy as np, itertools as itt
from pytriqs.operators import c as C, c_dag as CDag, n as N, dagger

from gfoperations import sum
from transformation import GfStructTransformationIndex


class Hubbard:
    """
    abstract class, realization needs self._c(s, i), self.sites, self.up, self.dn,
    self.gf_struct
    """
    def _c_dag(self, spin, site):
        return dagger(self._c(spin, site))

    def get_h_int(self):
        return np.sum([self.u * self._c_dag(self.up, i) * self._c(self.up, i) * self._c_dag(self.dn, i) * self._c(self.dn, i) for i in self.sites], axis = 0)

    def get_gf_struct(self):
        return self.gf_struct

    def n(self, s, i):
        return self._c_dag(s, i) * self._c(s, i)

    def n_tot(self):
        spins = [self.up, self.dn]
        return np.sum([self._c_dag(s, i) * self._c(s, i) for s, i in itt.product(spins, self.sites)])

    def nn(self, i, j):
        return self.n_per_site(i) * self.n_per_site(j)

    def ss(self, i, j):
        up = self.up
        dn = self.dn
        c = lambda b, i: self._c(b, i)
        cdag = lambda b, i: self._c_dag(b, i)
        op = 0
        op += .5 * cdag(up, i) * c(dn, i) * cdag(dn, j) * c(up, j)
        op += .5 * cdag(dn, i) * c(up, i) * cdag(up, j) * c(dn, j)
        op += self.szsz(i, j)
        return op

    def ss_tot(self):
        return np.sum([self.ss(i, j) for i, j in itt.product(*[self.sites]*2)])

    def sz(self, i):
        return .5 * (self.n(self.up, i) - self.n(self.dn, i))

    def szsz(self, i, j):
        return self.sz(i) * self.sz(j)

    def get_n_per_spin(self, s):
        return np.sum([self._c_dag(s, i) * self._c(s, i) for i in self.sites], axis = 0)

    def n_per_spin(self, s):
        return np.sum([self._c_dag(s, i) * self._c(s, i) for i in self.sites], axis = 0)

    def n_per_site(self, i):
        return np.sum([self._c_dag(s, i) * self._c(s, i) for s in self.spins], axis = 0)

    def get_n_tot(self):
        return np.sum([self.get_n_per_spin(s) for s in self.spins], axis = 0)


class Site(Hubbard):

    def __init__(self, u = None, spins = ["up", "dn"]):
        self.u = u
        self.spins = spins
        self.up = spins[0]
        self.dn = spins[1]
        self.gf_struct = [[self.up, range(1)], [self.dn, range(1)]]
        self.sites = range(1)

    def get_n_spin(self, spin):
        return CDag(spin, 0) * C(spin, 0)

    def get_n_tot(self):
        return np.sum([self.get_n_spin(s) for s in self.spins], axis = 0)

    def _c(self, s, i):
        return C(s, i)


class Plaquette(Hubbard):

    def __init__(self, u = None, spins = ["up", "dn"]):
        self.up = spins[0]
        self.dn = spins[1]
        self.sites = range(4)
        self.gf_struct = [[self.up, range(4)], [self.dn, range(4)]]


class PlaquetteMomentum(Hubbard):

    def __init__(self, u = None, spins = ["up", "dn"], momenta = ["G", "X", "Y", "M"], transformation = {"up": .5*np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]), "dn": .5*np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]])}):
        self.u = u
        self.up = spins[0]
        self.dn = spins[1]
        self.sites = range(4)
        self.spins = spins
        self.transformation = transformation
        self.block_labels = [spin+"-"+k for spin in spins for k in momenta]
        self.gf_struct = [[l, range(1)] for l in self.block_labels]
        self._to_mom = GfStructTransformationIndex(self.gf_struct, [[self.up, self.sites], [self.dn, self.sites]])
        #self.h_int = np.sum([u * self._c_dag(self.up, i) * self._c(self.up, i) * self._c_dag(self.dn, i) * self._c(self.dn, i) for i in self.sites], axis = 0)

    def _c(self, spin, site):
        return sum([self.transformation[spin][site, k_index] * C(*self._to_mom(spin, k_index)) for k_index in range(4)])


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


class HubbardTriangleMomentum:

    def __init__(self, u, spins, momenta, transformation):
        up = spins[0]
        dn = spins[1]
        self.spins = spins
        self.sites = range(3)
        self.transformation = transformation
        self.block_labels = [spin+"-"+k for spin in spins for k in momenta]
        self.gf_struct = [[l, range(1)] for l in self.block_labels]
        self._to_mom = GfStructTransformationIndex(self.gf_struct, [[up, self.sites], [dn, self.sites]])
        self.h_int = np.sum([u * self._c_dag(up, i) * self._c(up, i) * self._c_dag(dn, i) * self._c(dn, i) for i in self.sites], axis = 0)

    def _c(self, spin, site):
        return sum([self.transformation[spin][site, k_index] * C(*self._to_mom(spin, k_index)) for k_index in range(3)])

    def _c_dag(self, spin, site):
        return dagger(self._c(spin, site))

    def get_h_int(self):
        return self.h_int

    def get_gf_struct(self):
        return self.gf_struct

    def get_n_per_spin(self, spin):
        return np.sum([self._c_dag(spin, i) * self._c(spin, i) for i in self.sites], axis = 0)

    def get_n_tot(self):
        return np.sum([self.get_n_per_spin(s) for s in self.spins], axis = 0)
