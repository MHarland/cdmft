import itertools as itt, numpy as np
from pytriqs.operators import n as N, dagger, c as C
#from pytriqs.operators.util.hamiltonians import h_int_kanamori
from pytriqs.operators.util.U_matrix import U_matrix_kanamori

from bethe.triqs_mod.hamiltonians import h_int_kanamori


class Dimer:
    def __init__(self, u, j, spins = ['up', 'dn'], orbs = ['d', 'c'], sites = range(2), transf = None):
        self.u = u
        self.j = j
        self.spins = spins
        self.orbs = orbs
        self.sites = sites
        self.set_interaction(transf)
        self.gap_sz = None
        self.transf = transf

    def set_interaction(self, transf):
        self.transf = transf
        umat, upmat = U_matrix_kanamori(len(self.sites), self.u, self.j)
        mos = {(sn, on): (str(sn)+"-"+str(on), 0) for sn, on in itt.product(self.spins, self.orbs)}
        h = h_int_kanamori(self.spins, self.orbs, umat, upmat, self.j, map_operator_structure = mos, transf = transf)
        mos = {(sn, on): (str(sn)+"-"+str(on), 1) for sn, on in itt.product(self.spins, self.orbs)}
        h += h_int_kanamori(self.spins, self.orbs, umat, upmat, self.j, map_operator_structure = mos, transf = transf)
        self.h_int = h

    def get_h_int(self):
        return self.h_int

    def get_gf_struct(self):
        return [(s+'-'+orb, self.sites) for s, orb in itt.product(self.spins, self.orbs)]

    def get_field_sz(self, gap):
        up, dn = self.spins[0], self.spins[1]
        field = .5 * gap * np.sum([self.n(self.spins[1], o, i) for o, i in itt.product(self.orbs, self.sites)], axis = 0)
        field -= .5 * gap * np.sum([self.n(self.spins[0], o, i) for o, i in itt.product(self.orbs, self.sites)], axis = 0)
        return field
    
    def add_field_sz(self, gap):
        if self.gap_sz is None:
            self.h_int += self.get_field_sz(gap)
            self.gap_sz = gap

    def rm_field_sz(self):
        if self.gap_sz is not None:
            self.h_int -= self.get_field_sz(self.gap_sz)
            self.gap_sz = None

    def c(self, spin, orb, site):
        """
        assumes spin-orb blockstructure
        """
        block = spin+'-'+orb
        if self.transf is None:
            cnew = C(block, site)
        else:
            sites = range(self.transf[block].shape[0])
            cnew =  np.sum([self.transf[block][site, i] * C(orb, i) for i in sites], axis = 0)
        return cnew

    def c_dag(self, spin, orb, site):
        return dagger(self.c(spin, orb, site))

    def n(self, spin, orb, site):
        return self.c_dag(spin, orb, site) * self.c(spin, orb, site)

    def n_tot(self):
        return np.sum([self.n(s, o, i) for s, o, i in itt.product(self.spins, self.orbs, self.sites)], axis = 0)
