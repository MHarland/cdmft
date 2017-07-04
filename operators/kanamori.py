import itertools as itt, numpy as np
from pytriqs.operators import n as N, dagger, c as C


class Dimer:
    def __init__(self, u = None, j = None, spins = ['up', 'dn'], orbs = ['d', 'c'], sites = range(2), transf = None, density_density_only = False):
        self.u = u
        self.j = j
        self.spins = spins
        self.orbs = orbs
        self.sites = sites
        self.gap_sz = None
        self.transf = transf
        if transf:
            self.kinds = range(self.transf.values()[0].shape[0])
        if u is not None and j is not None: self.set_h_int(u, j, density_density_only)

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
            self.h_int += gap * self.sz_tot()
            self.gap_sz = gap
        else:
            self.rm_field_sz()
            self.h_int += gap * self.sz_tot()
            self.gap_sz = gap

    def rm_field_sz(self):
        if self.gap_sz is not None:
            self.h_int -= self.gap_sz * self.sz_tot()
            self.gap_sz = None

    def c(self, spin, orb, site):
        """
        assumes spin-orb blockstructure
        """
        block = spin+'-'+orb
        if self.transf is None:
            cnew = C(block, site)
        else:
            cnew =  np.sum([self.transf[block][k, site].conjugate() * C(block, k) for k in self.kinds], axis = 0)
        return cnew

    def c_dag(self, spin, orb, site):
        return dagger(self.c(spin, orb, site))

    def n(self, spin, orb, site):
        return self.c_dag(spin, orb, site) * self.c(spin, orb, site)

    def n_tot(self):
        return np.sum([self.n(s, o, i) for s, o, i in itt.product(self.spins, self.orbs, self.sites)], axis = 0)

    def n_per_spin(self, spin):
        return np.sum([self.n(spin, o, i) for o, i in itt.product(self.orbs, self.sites)], axis = 0)

    def sz_tot(self):
        return .5 * (self.n_per_spin(self.spins[0]) - self.n_per_spin(self.spins[1]))

    def sz2_tot(self):
        return self.sz_tot() * self.sz_tot()

    def set_h_int(self, u, j, density_density_only = False):
        self.up = u - 2 * j
        self.u = u
        self.j = j
        self.h_int = np.sum([self.h_int_per_site(i, density_density_only) for i in self.sites], axis = 0)

    def h_int_per_site(self, site, density_density_only = False):
        uintra = self.u * np.sum([self.n(self.spins[0], o, site) * self.n(self.spins[1], o, site) for o in self.orbs], axis = 0)
        uinter = self.up * np.sum([self.n(s1, self.orbs[0], site) * self.n(s2, self.orbs[1], site) for s1, s2 in itt.product(self.spins, self.spins)], axis = 0)
        jpara = -self.j * np.sum([self.n(s, self.orbs[0], site) * self.n(s, self.orbs[1], site) for s in self.spins], axis = 0)
        if not density_density_only:
            cross = np.sum([self.c_dag(self.spins[1], o1, site) * self.c_dag(self.spins[0], o2, site) * self.c(self.spins[1], o2, site) * self.c(self.spins[0], o1, site)
                            + self.c_dag(self.spins[0], o2, site) * self.c_dag(self.spins[1], o2, site) * self.c(self.spins[0], o1, site) * self.c(self.spins[1], o1, site)
                            for o1, o2 in itt.product(self.orbs, self.orbs) if o1 != o2], axis = 0)
            jortho = -.5 * self.j * (cross + dagger(cross))
        else:
            jortho = 0
        return uintra + uinter + jpara + jortho

    def sz(self, orb, site):
        return .5 * (self.n(self.spins[0], orb, site) - self.n(self.spins[1], orb, site))

    def s_plus(self, orb, site):
        return self.c_dag(self.spins[0], orb, site) * self.c(self.spins[1], orb, site)

    def s_minus(self, orb, site):
        return self.c_dag(self.spins[1], orb, site) * self.c(self.spins[0], orb, site)

    def s2_tot(self):
        return np.sum([.5 * (self.s_plus(o1, i1) * self.s_minus(o2, i2) + self.s_minus(o1, i1) * self.s_plus(o2, i2)) + self.sz(o1, i1) * self.sz(o2, i2) for i1, i2, o1, o2 in itt.product(self.sites, self.sites, self.orbs, self.orbs)], axis = 0)


class MomentumDimer(Dimer):
    """
    assumes spin-orb-mom blockstructure
    """
    def __init__(self, *args, **kwargs):
        self.mom = kwargs.pop('momenta')
        self.r_to_k = {i:self.mom[i] for i in range(len(self.mom))}
        Dimer.__init__(self, *args, **kwargs)
    
    def c(self, spin, orb, site):
        block = spin+'-'+orb
        c =  np.sum([self.transf[block][i_k, site].conjugate() * C(block+'-'+self.r_to_k[i_k], 0) for i_k in self.kinds], axis = 0)
        return c
