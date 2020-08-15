import numpy as np, itertools as itt

from cdmft.transformation import MatrixTransformation


class LatticeDispersion:
    """
    hopping is a dict with numpy vectors in the lattice basis as keys
    only for orthogonal lattice vectors, so far
    """
    def __init__(self, hopping, k_points_per_dimension, spins = ['up', 'dn']):
        self.spins = spins
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
        self.energies_k = self.energies
        self.energies_spinsite_space = np.array([{s: h for s in self.spins} for h in self.energies])
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
        self.energies = np.array([np.sum([t * np.exp(complex(0,-2*np.pi*k.dot(r))) for t, r in itt.izip(self.hopping_elements, self.translations)], axis = 0) for k, w in itt.izip(self.bz_points, self.bz_weights)])

    def loop_over_bz(self):
        for k, w, d in itt.izip(self.bz_points, self.bz_weights, self.energies):
            yield k, w, d

    def transform(self, matrixtransformation):
        for i, d in enumerate(self.energies):
            self.energies[i] = matrixtransformation.transform(d)

    def transform_site_space(self, unitary_transformation_matrix, new_blockstructure = None, reblock_map = None):
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
        if new_blockstructure is None:
            site_transf = MatrixTransformation(site_struct, unitary_transformation_matrix,
                                               site_struct)
        else:
            site_transf = MatrixTransformation(site_struct, unitary_transformation_matrix,
                                               new_blockstructure)
        for i, d in enumerate(self.energies_spinsite_space):
            self.energies[i] = site_transf.transform_matrix(d)


class SquarelatticeDispersion(LatticeDispersion):
    """
    TODO needs careful testing, not sure whether phase is correct; don't use!
    It's probably easier and safer to rewrite this class
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


class LatticeDispersionMultiband(LatticeDispersion):
    def __init__(self, orb_disp_map = {}):
        self.energies = []
        more_k_to_iterate = True
        i_k = 0
        while more_k_to_iterate:
            self.energies.append(dict([(orbname, orbdisp.energies_k[i_k]) for orbname, orbdisp in orb_disp_map.items()]))
            for orbdisp in orb_disp_map.values():
                if len(orbdisp.energies_k) - 1 == i_k:
                    more_k_to_iterate = False
                    break
            i_k += 1
        self.struct = [[key, range(len(val))] for key, val in self.energies[0].items()]
        self.energies = np.array(self.energies)
        for orbdisp in orb_disp_map.values():
            self.bz_points = orbdisp.bz_points
            self.bz_weights = orbdisp.bz_weights
            break

    def transform_site_space(self, unitary_transformation_matrix, orbital_filter = [], new_blockstructure = None, reblock_map = None):
        """
        uses the orbital_filter option of MatrixTransformation
        """
        site_struct = self.struct
        if new_blockstructure is None:
            site_transf = MatrixTransformation(site_struct, unitary_transformation_matrix,
                                               site_struct, orbital_filter = orbital_filter)
        else:
            site_transf = MatrixTransformation(site_struct, unitary_transformation_matrix,
                                               new_blockstructure, orbital_filter = orbital_filter)
        for i, d in enumerate(self.energies):
            self.energies[i] = site_transf.transform_matrix(d)
