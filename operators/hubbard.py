import numpy as np, itertools as itt
from scipy.linalg import expm, inv
from pytriqs.operators import c as C, c_dag as CDag, n as N, dagger

from bethe.gfoperations import sum
from bethe.transformation import GfStructTransformationIndex


class Hubbard:
    """
    meant as abstract class, realization needs self._c(s, i), self.sites, self.up, self.dn,
    self.spins, self.u
    """
    def _c(self, spin, site, *args, **kwargs):
        return C(spin, site)

    def _c_dag(self, spin, site, *args, **kwargs):
        return dagger(self._c(spin, site, *args, **kwargs))

    def get_h_int(self):
        """for (C)DMFT calculations"""
        return np.sum([self.u * self._c_dag(self.up, i) * self._c(self.up, i) * self._c_dag(self.dn, i) * self._c(self.dn, i) for i in self.sites], axis = 0)

    def h_int_cluster(self, t, mu):
        spins = [self.up, self.dn]
        return self.get_h_int() +  self.kinetic_energy(t) - np.sum([self._c_dag(s, i) * mu * self._c(s, i) for s, i in itt.product(spins, self.sites)], axis = 0)

    def kinetic_energy(self, t):
        spins = [self.up, self.dn]
        return np.sum([self._c_dag(s, i) * t[s][i, j] * self._c(s, j) for s, i, j in itt.product(spins, self.sites, self.sites)], axis = 0)

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

    def s_plus(self, i, j):
        return self._c_dag(self.up, i) * self._c(self.dn, j)

    def s_minus(self, i, j):
        return self._c_dag(self.dn, i) * self._c(self.up, j)
    
    def ss_pm_loc(self, site):
        return self.s_plus(site, site) * self.s_minus(site, site)

    def ss_mp_loc(self, site):
        return self.s_minus(site, site) * self.s_plus(site, site)

    def ss_tot(self):
        return np.sum([self.ss(i, j) for i, j in itt.product(*[self.sites]*2)])

    def nn_tot(self):
        return np.sum([self.nn(i, j) for i, j in itt.product(*[self.sites]*2)])

    def sz(self, i):
        return .5 * (self.n(self.up, i) - self.n(self.dn, i))

    def sz_tot(self):
        return np.sum([self.sz(i) for i in self.sites])

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
            c = np.sum([self.transf[s][j, i].conjugate() * C(s, j) for j in self.sites])
        return c


class Triangle(Hubbard):

    def __init__(self, u = None, spins = ["up", "dn"], transf = None):
        self.u = u
        self.spins = spins
        self.up = spins[0]
        self.dn = spins[1]
        self.sites = range(3)
        self.transf = transf

    def _c(self, s, i):
        if self.transf is None:
            c = Hubbard._c(self, s, i)
        else:
            c = np.sum([self.transf[s][j, i].conjugate() * C(s, j) for j in self.sites])
        return c


class TriangleSpinOrbitCoupling(Triangle):
    """
    1 block, spin-site blockstructure
    transformation is not a dict, but an array acting on the site-space
    the aiao-field: the rotation MUST be applied first since it depends non-linearly on the 
    site-space, the site-transformation comes second
    """
    def __init__(self, blocklabel, *args, **kwargs):
        Triangle.__init__(self, *args, **kwargs)
        self.blocklabel = blocklabel
        self.blocksize = len(self.sites) * len(self.spins)

    def _c(self, s, i, theta = 0, phi = 0):
        if self.transf is None:
            c = C(self.blocklabel, self.superindex(s, i))
        else:
            c = np.sum([self.transf[s][j, i].conjugate() * C(self.blocklabel, self.superindex(s, j)) for j in self.sites], axis = 0)
        return c

    def spin_index(self, s):
        return {self.spins[0]: 0, self.spins[1]: 1}[s]

    def _c_rot(self, s, i, theta, phi = 0):
        spin_transf_mat = self.spin_transf_mat(theta, phi)
        c = np.sum([spin_transf_mat[self.spin_index(s), self.spin_index(t)] * self._c(t, i) for t in self.spins], axis = 0)
        return c

    def _c_rot_dag(self, s, i, theta, phi = 0):
        return dagger(self._c_rot(s,i,theta,phi))
    
    def spin_transf_mat(self, theta, phi = 0, force_real = True):
        py = np.matrix([[0,complex(0,-1)],[complex(0,1),0]])
        pz = np.matrix([[1,0],[0,-1]])
        m = expm(complex(0,-1)*theta*py*.5)#.dot(expm(complex(0,1)*phi*pz*.5))
        if force_real:
            m = m.real
        return m

    def aiao_op(self, chirality = [0, 1, 2]):
        """
        chiralities are either 0,1,2 or 0,2,1
        """
        operator = 0
        phi = 0
        for i in self.sites:
            theta = chirality[i] * 2 * np.pi / 3.
            for s, sign in zip(self.spins, [+1, -1]):
                operator += sign * self._c_rot_dag(s, i, theta, phi) * self._c_rot(s, i, theta, phi)
        return operator

    def superindex(self, s, i):
        if s in self.spins:
            s = self.spin_index(s)
        return s * 3 + i


class TriangleAIAO(Triangle):
    """

    """
    def __init__(self, *args, **kwargs):
        self.theta = kwargs.pop('theta') if 'theta' in kwargs.keys() else 0
        self.phi = kwargs.pop('phi') if 'phi' in kwargs.keys() else 0
        self.force_real = kwargs.pop('force_real') if 'force_real' in kwargs.keys() else False
        self.site_transf = kwargs.pop('site_transf') if 'site_transf' in kwargs.keys() else False
        Triangle.__init__(self, *args, **kwargs)

    def _c(self, s, i):
        c = 0
        if self.site_transf:
            for j in range(3):
                c += self.site_transf[j, i].conjugate() * self._c_rot(s, j)
        else:
            c = self._c_rot(s, j)
        return c

    def _c_rot(self, s, i):
        s = self.spin_index(s)
        c = 0
        for t in range(2):
            a = self.superindex(t, i)
            c += inv(self.spin_transf_mat(self.theta, self.phi))[s, t] * C('spin-site', a)
        return c

    def superindex(self, s, i):
        if s in self.spins:
            s = self.spin_index(s)
        return s * 3 + i

    def spin_index(self, s):
        return {'up':0, 'dn':1}[s]

    def spin_transf_mat(self, theta, phi = 0):
        py = np.matrix([[0,complex(0,-1)],[complex(0,1),0]])
        pz = np.matrix([[1,0],[0,-1]])
        m = expm(complex(0,-1)*theta*py*.5)#.dot(expm(complex(0,1)*phi*pz*.5))
        if self.force_real:
            m = m.real
        return m


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
        return sum([self.transformation[spin][k_index, site].conjugate() * C(*self._to_mom(spin, k_index)) for k_index in range(len(self.sites))])



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
        return sum([self.transformation[spin][k_index, site].conjugate() * C(*self._to_mom(spin, k_index)) for k_index in range(len(self.sites))])

    def doublet_state(self, i, j, sz, pm = -1):
        for site in self.sites:
            if not (site in [i, j]):
                k = site
        return (self._c(self.up, i) * self._c(self.dn, j) +pm* self._c(self.dn, i) * self._c(self.up, j)) * self._c(sz, k) / np.sqrt(2)

    def nn_singlet_n2_state(self, i, j, pm = -1):
        return (self._c(self.up, i) * self._c(self.dn, j) +pm* self._c(self.dn, i) * self._c(self.up, j)) / np.sqrt(2)

    def nn_singlet_n4_state(self, i, j, pm = -1):
        for site in self.sites:
            if not (site in [i, j]):
                k = site
        return (self._c(self.up, i) * self._c(self.dn, j) +pm* self._c(self.dn, i) * self._c(self.up, j)) * self._c(self.up, k) * self._c(self.dn, k)/ np.sqrt(2)

    def rvb_projector(self, particle_numbers = [2,3,4], pm = -1):
        inds = [(i, j) for i in self.sites for j in range(i)]#itt.product(self.sites, self.sites)]
        terms = []
        for i in inds:
            if 2 in particle_numbers:
                terms.append(self.nn_singlet_n2_state(*i, pm = pm))
            if 3 in particle_numbers:
                terms.append(self.doublet_state(i[0], i[1], self.up, pm = pm)+self.doublet_state(i[0], i[1], self.dn, pm = pm))
            if 4 in particle_numbers:
                terms.append(self.nn_singlet_n4_state(*i, pm = pm))
        state = np.sum(terms, axis = 0)
        return dagger(state) * state



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
        self.momenta = momenta
        self.transformation = transformation
        self.block_labels = [spin+"-"+k for spin in spins for k in momenta]
        self.gf_struct = [[l, range(1)] for l in self.block_labels]
        self._to_mom = GfStructTransformationIndex(self.gf_struct, [[self.up, self.sites], [self.dn, self.sites]])

    def _c(self, spin, site):
        return sum([self.transformation[spin][site, k_index] * C(*self._to_mom(spin, k_index)) for k_index in range(4)])

    def cdup_cup_cddn_cdn(self, i, j, k, l):
        """i,j,k,l being momenta"""
        return CDag(self.up+'-'+i,0) * C(self.up+'-'+j,0) * CDag(self.dn+'-'+k,0) * C(self.dn+'-'+l,0)


class PlaquetteMomentumNambu(Hubbard):
    """
    extends PlaquetteMomentum space by anomalous parts, using particle-hole transformation on 
    spin-down
    """
    def __init__(self, u, spins, momenta, transformation):
        self.u = u
        self.sites = range(4)
        self.up, self.dn = up, dn = spins[0], spins[1]
        self.site_to_mom = dict([(i, momenta[i]) for i in range(4)])
        self.transformation = transformation
        self.block_labels = [k for k in momenta]
        self.gf_struct = [[l, range(2)] for l in self.block_labels]

    def _c(self, spin, site):
        if spin == self.up:
            return sum([self.transformation[spin][k_index, site].conjugate() * C(self.site_to_mom[k_index], 0) for k_index in range(4)])
        elif spin == self.dn:
            return sum([self.transformation[spin][k_index, site].conjugate() * CDag(self.site_to_mom[k_index], 1) for k_index in range(4)]) # TODO what's first, momentum or nambu transf?
        assert False, "spin "+spin+" not recognized"


class PlaquetteMomentumAFMNambu(Hubbard):
    """
    adds afm
    """
    def __init__(self, u, spins, momenta, transformation):
        self.u = u
        self.sites = range(4)
        self.up, self.dn = up, dn = spins[0], spins[1]
        self.site_to_mom_up = {0: ("GM", 0), 1: ("GM", 2), 2: ("XY", 0), 3: ("XY", 2)}
        self.site_to_mom_dn = {0: ("GM", 1), 1: ("GM", 3), 2: ("XY", 1), 3: ("XY", 3)}
        self.transformation = transformation
        self.block_labels = [k for k in momenta]
        self.gf_struct = [[l, range(2)] for l in self.block_labels]

    def _c(self, spin, site):
        if spin == self.up:
            return sum([self.transformation[spin][k_index, site].conjugate() * C(*self.site_to_mom_up[k_index]) for k_index in range(4)])
        elif spin == self.dn:
            return sum([self.transformation[spin][k_index, site].conjugate() * CDag(*self.site_to_mom_dn[k_index]) for k_index in range(4)])
        assert False, "spin "+spin+" not recognized"

    def cdup_cup_cddn_cdn(self, i, j, k, l):
        """i,j,k,l being momenta"""
        return  CDag(*self.site_to_mom_up[i]) * C(*self.site_to_mom_up[j]) * C(*self.site_to_mom_dn[k]) * CDag(*self.site_to_mom_dn[l])
