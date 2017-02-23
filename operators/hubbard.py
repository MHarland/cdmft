import numpy as np, itertools as itt
from pytriqs.operators import c as C, c_dag as CDag, n as N, dagger

from bethe.gfoperations import sum
from bethe.transformation import GfStructTransformationIndex


class Hubbard:
    """
    meant as abstract class, realization needs self._c(s, i), self.sites, self.up, self.dn,
    self.spins, self.u
    """
    def _c(self, spin, site):
        return C(spin, site)

    def _c_dag(self, spin, site):
        return dagger(self._c(spin, site))

    def get_h_int(self):
        return np.sum([self.u * self._c_dag(self.up, i) * self._c(self.up, i) * self._c_dag(self.dn, i) * self._c(self.dn, i) for i in self.sites], axis = 0)

    def get_gf_struct(self):
        return [[self.up, self.sites], [self.dn, self.sites]]

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

    def nn_tot(self):
        return np.sum([self.nn(i, j) for i, j in itt.product(*[self.sites]*2)])

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
        self.sites = range(1)


class Dimer(Hubbard):
    """
    support unitary transformation on site-space U: c -> U.c, implying e.g. G -> U.G.U^dag
    but no reblocking and relabeling so far # TODO
    """
    def __init__(self, u = None, spins = ["up", "dn"], transf = None):
        self.u = u
        self.spins = spins
        self.up = spins[0]
        self.dn = spins[1]
        self.sites = range(2)
        self.transf = transf
 
    def _c(self, s, i):
        if self.transf is None:
            c = Hubbard._c(self, s, i)
        else:
            c = np.sum([self.transf[i, j] * C(s, j) for j in self.sites])
        return c


class Triangle(Hubbard):

    def __init__(self, u = None, spins = ["up", "dn"]):
        self.u = u
        self.spins = spins
        self.up = spins[0]
        self.dn = spins[1]
        self.sites = range(3)


class Plaquette(Hubbard):

    def __init__(self, u = None, spins = ["up", "dn"]):
        self.u = u
        self.spins = spins
        self.up = spins[0]
        self.dn = spins[1]
        self.sites = range(4)


class DimerMomentum(Hubbard):
    """
    transformation is a 2by2 matrix applied to the site-space
    """
    def __init__(self, u = None, spins = ["up", "dn"], momenta = ["+", "-"], transformation = {"up": np.sqrt(.5)*np.array([[1,1],[1,-1]]), "dn": np.sqrt(.5)*np.array([[1,1],[1,-1]])}):
        self.u = u
        self.up = spins[0]
        self.dn = spins[1]
        self.spins = spins
        self.transformation = transformation
        self.sites = range(2)
        self.block_labels = [spin+"-"+k for spin in spins for k in momenta]
        self.gf_struct = [[l, range(1)] for l in self.block_labels]
        self._to_mom = GfStructTransformationIndex(self.gf_struct, [[self.up, self.sites], [self.dn, self.sites]])

    def _c(self, spin, site):
        return sum([self.transformation[spin][site, k_index] * C(*self._to_mom(spin, k_index)) for k_index in range(len(self.sites))])



class TriangleMomentum(Hubbard):
    """
    transformation is a 3by3 matrix applied to the site-space
    """
    def __init__(self, u = None, spins = ["up", "dn"], momenta = ["E", "A1", "A2"], transformation = {"up": np.array([[1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],[0,-1/np.sqrt(2),1/np.sqrt(2)],[-np.sqrt(2./3.),1/np.sqrt(6),1/np.sqrt(6)]]), "dn": np.array([[1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],[0,-1/np.sqrt(2),1/np.sqrt(2)],[-np.sqrt(2./3.),1/np.sqrt(6),1/np.sqrt(6)]])}):
        self.u = u
        self.up = spins[0]
        self.dn = spins[1]
        self.spins = spins
        self.transformation = transformation
        self.sites = range(3)
        self.block_labels = [spin+"-"+k for spin in spins for k in momenta]
        self.gf_struct = [[l, range(1)] for l in self.block_labels]
        self._to_mom = GfStructTransformationIndex(self.gf_struct, [[self.up, self.sites], [self.dn, self.sites]])

    def _c(self, spin, site):
        return sum([self.transformation[spin][k_index, site] * C(*self._to_mom(spin, k_index)) for k_index in range(len(self.sites))]) # TODO conjugate!


class TriangleMomentum2(TriangleMomentum):
    def _c(self, spin, site):
        return sum([self.transformation[spin][k_index, site] * C(*self._to_mom(spin, k_index)) for k_index in range(len(self.sites))])
    

class PlaquetteMomentum(Hubbard):
    """
    transformation is a 4by4 matrix applied to the site-space
    """
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

    def _c(self, spin, site):
        return sum([self.transformation[spin][site, k_index] * C(*self._to_mom(spin, k_index)) for k_index in range(4)])


class PlaquetteMomentumNambu(Hubbard): # TODO
    """
    extends PlaquetteMomentum space by anomalous parts, using particle-hole transformation on 
    spin-down
    """
    def __init__(self, u, spins, momenta, transformation):
        self.u = u
        up = spins[0]
        dn = spins[1]
        self.up = up
        self.dn = dn
        sites = range(4)
        self.site_to_mom = dict([(i, momenta[i]) for i in sites])
        self.transformation = transformation
        self.block_labels = [k for k in momenta]
        self.gf_struct = [[l, range(2)] for l in self.block_labels]

    def _c(self, spin, site):
        if spin == self.up:
            return sum([self.transformation[spin][site, k_index] * C(self.site_to_mom[k_index], 0) for k_index in range(len(self.sites))])
        elif spin == self.dn:
            return sum([self.transformation[spin][site, k_index] * CDag(self.site_to_mom[k_index], 1) for k_index in range(len(self.sites))])
        assert False, "spin "+spin+" not recognized"
