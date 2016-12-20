import numpy as np, itertools as itt

from bethe.transformation import MatrixTransformation


class LatticeDispersion:
    """
    hopping is a dict with numpy vectors in the lattice basis as keys
    """
    def __init__(self, hopping, k_points_per_dimension, spins = ['up', 'dn'], force_real = True):
        self.spins = spins
        self.force_real = force_real
        for r, t in hopping.items():
            self.dimension = len(r)
            self.n_orbs = len(t)
            break
        rs = []
        t_rs = []
        for r, t in hopping.items():
            rs.append(np.array(r))
            t_rs.append(np.array(t))
        self.translations = np.array(rs)
        self.hopping_elements = np.array(t_rs)
        self.create_grid(k_points_per_dimension)
        self.calculate_energies()
        up, dn = spins[0], spins[1]
        self.energies_spinsite_space = np.array([{up: h, dn: h} for h in self.energies])
        self.energies = self.energies_spinsite_space

    def create_grid(self, k_points_per_dimension):
        bz_points_per_dim = np.linspace(-.5, .5, k_points_per_dimension, False)
        self.bz_points = []
        for k in itt.product(*[bz_points_per_dim] * self.dimension):
            self.bz_points.append(k)
        self.bz_points = np.array(self.bz_points)
        n_k = self.bz_points.shape[0]
        self.bz_weights = np.array([1./n_k] * n_k)

    def calculate_energies(self):
        if self.force_real:
            self.energies = np.array([np.sum([t * np.exp(complex(0,-2*np.pi*k.dot(r))) for t, r in itt.izip(self.hopping_elements, self.translations)], axis = 0).real for k, w in itt.izip(self.bz_points, self.bz_weights)])
        else:
            self.energies = np.array([np.sum([t * np.exp(complex(0,-2*np.pi*k.dot(r))) for t, r in itt.izip(self.hopping_elements, self.translations)], axis = 0) for k, w in itt.izip(self.bz_points, self.bz_weights)])

    def loop_over_bz(self):
        for k, w, d in itt.izip(self.bz_points, self.bz_weights, self.energies):
            yield k, w, d

    def transform_site_space(self, unitary_transformation_matrix, new_blockstructure, reblock_map):
        """
        assumes that all subspin orbitals are site degrees of freedom, i.e. single orbital setup
        unitary_transformation_matrix must be a dict with up and dn keys and matrix values
        describing the transformation on site space
        new_blockstructure is a list of lists, where a sublist has two entries, first the new
        blockname and second range(nr of orbitals of this block)
        reblock_map maps 3-tuples of the old(spin-site) structure to the new_blockstructure
        side-note: this mapping need not be invertible
        """
        site_struct = [[s, range(self.n_orbs)] for s in self.spins]
        site_transf = MatrixTransformation(site_struct, unitary_transformation_matrix,
                                           new_blockstructure)
        for i, d in enumerate(self.energies_spinsite_space):
            self.energies[i] = site_transf.reblock_by_map(site_transf.transform_matrix(d),
                                                          reblock_map)


class SquarelatticeDispersion(LatticeDispersion):
    """
    Uses irreducible wedge
    """
    pass
