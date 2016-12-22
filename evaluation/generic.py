import numpy as np, itertools as itt
from pytriqs.applications.impurity_solvers.cthyb import AtomDiag, atomic_density_matrix
from pytriqs.gf.local import BlockGf, GfImTime


class Evaluation:
    
    def __init__(self, archive):
        self.archive = archive
        self.n_loops = self.archive.get_completed_loops()

    def get_sign_loop(self):
        signs = np.empty([self.n_loops]*2)
        for i in range(self.n_loops):
            signs[i, 0] = i
            signs[i, 1] = self.archive.load("average_sign", i)
        return signs

    def get_density(self, loop = -1):
        return self.archive.load("density", loop)

    def get_density_matrix(self, loop = -1):
        rhoblocked = self.archive.load("density_matrix", loop, bcast = False)
        atom = self.archive.load("h_loc_diagonalization", loop, bcast = False)
        rho = np.zeros([atom.full_hilbert_space_dim]*2)
        for i_block, block in enumerate(rhoblocked):
            dim = atom.get_block_dim(i_block)
            for i, j in itt.product(*[range(dim)]*2):
                k = atom.flatten_block_index(i_block, i)
                l = atom.flatten_block_index(i_block, j)
                rho[k, l] = block[i, j]
        return rho
        
    def get_energies(self, loop = -1):
        atom = self.archive.load("h_loc_diagonalization", loop, bcast = False)
        energies = []
        for energy_block in atom.energies:
            for i in range(len(energy_block)):
                energies.append(energy_block[i])
        return np.array(energies)

    def get_density_matrix_diag(self, loop = -1):
        """order corresponds to energies of get_energies"""
        rho = self.archive.load("density_matrix", loop, bcast = False)
        probabilities = []
        for rho_block in rho:
            for i in range(len(rho_block)):
                probabilities.append(rho_block[i, i])
        return np.array(probabilities)

    def get_density_matrix_row(self, row, loop = -1):
        """order corresponds to energies of get_energies"""
        rhoblocked = self.archive.load("density_matrix", loop, bcast = False)
        atom = self.archive.load("h_loc_diagonalization", loop, bcast = False)
        rhorow = np.zeros([atom.full_hilbert_space_dim])
        energies = np.zeros([atom.full_hilbert_space_dim])
        for i_block, block in enumerate(rhoblocked):
            dim = atom.get_block_dim(i_block)
            for j in range(dim):
                if atom.flatten_block_index(i_block, j) == row:
                    row_block = i_block
                    row_block_j = j
        for j in range(atom.get_block_dim(row_block)):
            number = rhoblocked[row_block][row_block_j, j]
            if number < 0:
                print "warning, taking absolute value"
            rhorow[atom.flatten_block_index(row_block, 0)+j] = abs(number)
        return rhorow

    def get_atomic_density_matrix_diag(self, loop = -1, beta = None):
        """order corresponds to energies of get_energies"""
        atom = self.archive.load("h_loc_diagonalization", loop, bcast = False)
        if beta is None:
            g = self.archive.load("g_loc_iw", loop)
            beta = g.beta
        rho = atomic_density_matrix(atom, beta)
        probabilities = []
        for rho_block in rho:
            for i in range(len(rho_block)):
                probabilities.append(rho_block[i, i])
        return np.array(probabilities)

    def get_g_static_diags(self, loop = -1):
        g = self.archive.load("g_imp_iw")
        gf_struct = []
        for s, b in g:
            gf_struct.append([s, range(len(b.indices))])
        gtau = BlockGf(name_block_generator = [(struct[0], GfImTime(indices = struct[1], beta = g.beta, n_points = 10001)) for struct in gf_struct])
        for s, b in gtau: b.set_from_inverse_fourier(g[s])
        inds = [i for i in g.all_indices]
        gsd = {}
        for ind in inds:
            if ind[1] == ind[2]:
                i = int(ind[1])
                b = ind[0]
                gsd[b+'-'+str(i)+str(i)] = -gtau[b].data[-1, i, i].real
        return gsd
