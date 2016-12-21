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
            self.dimension = len(r) if type(r) != int else 1
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
            self.energies = np.array([np.sum([t * np.exp(complex(0,-2*np.pi*k.dot(r))) for t, r in itt.izip(self.hopping_elements, self.translations)], axis = 0).real for k in self.bz_points])
        else: # TODO
            self.energies = np.array([np.sum([t * np.exp(complex(0,-2*np.pi*k.dot(r))) for t, r in itt.izip(self.hopping_elements, self.translations)], axis = 0) for k, w in itt.izip(self.bz_points, self.bz_weights)])

    def loop_over_bz(self):
        for k, w, d in itt.izip(self.bz_points, self.bz_weights, self.energies):
            yield k, w, d

    def transform_site_space(self, unitary_transformation_matrix, new_blockstructure, reblock_map = None):
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
            self.energies[i] = site_transf.transform_matrix(d)


class SquarelatticeDispersion(LatticeDispersion):
    """
    Uses irreducible wedge
    TODO easy to generalize, hard to optimize
    """
    def create_grid(self, *args, **kwargs):
        LatticeDispersion.create_grid(self, *args, **kwargs)
        rotate = lambda phi: np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        #symmetry_ops = [rotate(i * 2 * np.pi / 4.) for i in range(4)] + [np.array([[-1,0],[0,1]])]
        symmetry_ops = [np.array([[-1,0],[0,1]])]
        self.equivalent_k = [[k] for k in self.bz_points]
        for op in symmetry_ops: # TODO c++
            for k in self.bz_points:
                merged = self.merge_classes(k, op.dot(k), self.equivalent_k)
        self.bz_weights = np.array([len(kclass)/float(len(self.bz_points)) for kclass in self.equivalent_k])
        self.bz_points = np.array([kclass[0] for kclass in self.equivalent_k])

    def merge_classes(self, el1, el2, classes):
        """
        adds x to class of y
        """
        merged = False
        i_el1 = self.find_class(el1, classes)
        if i_el1 is not None and i_el1 != self.find_class(el2, classes):
            class1 = classes[i_el1]
            del classes[i_el1]
            i_el2 = self.find_class(el2, classes)
            if i_el2 is not None:
                class2 = classes[i_el2]
                del classes[i_el2]
                classes.append(class1 + class2)
                mergend = True
            else:
                classes.append(class1)
        return merged

    def find_class(self, element, classes):
        for i, aclass in enumerate(classes):
            for el in aclass:
                if np.allclose(el, element):
                    return i
        return None

    def calculate_energies(self):
        self.energies = []
        for kclass in self.equivalent_k:
            term = 0
            for t, r in itt.izip(self.hopping_elements, self.translations):
                #term += np.sum([t * np.exp(complex(0,-2*np.pi*k.dot(r))) for k in kclass], axis = 0)  / len(kclass)
                term += t * np.exp(complex(0,-2*np.pi*kclass[0].dot(r)))
            if self.force_real:
                term = term.real
            self.energies.append(term)


class SquarelatticeDispersionFast(LatticeDispersion):
    """
    not tested, don't use it!
    works only for single site (?)
    Uses irreducible wedge
    specialized, faster
    needs diagonal/clustermomenta basis
    """
    def create_grid(self, k_points_per_dimension):
        assert k_points_per_dimension % 2 == 0, "number of k-points of a dimension must be even using the irreducible wedge of the squarelattice"
        self.bz_points = []
        self.bz_weights = []
        n = k_points_per_dimension / 2 + 1
        unit_weight = 1. / k_points_per_dimension**2
        step = .5 / (n - 1)
        for i in range(n):
            if i == 0:
                for j in range(n):
                    if j == 0:
                        self.bz_weights.append(1 * unit_weight)
                    elif j == (n - 1):
                        self.bz_weights.append(2 * unit_weight)
                    else:
                        self.bz_weights.append(4 * unit_weight)
                    self.bz_points.append([-.5 + j * step, -.5])
            elif i == n - 1:
                self.bz_weights.append(1 * unit_weight)
                self.bz_points.append([0, 0])
            else:
                for j in range(i, n):
                    if j == i or j == n - 1:
                        self.bz_weights.append(4 * unit_weight)
                    else:
                        self.bz_weights.append(8 * unit_weight)
                    self.bz_points.append([-.5 + i * step, -.5 + j * step])
        self.bz_points = np.array(self.bz_points)
        self.bz_weights = np.array(self.bz_weights)
