import numpy as np
from pytriqs.applications.impurity_solvers.cthyb import AtomDiag, trace_rho_op, act
from pytriqs.operators import c as C, c_dag as CDag, n as N


class StaticObservable:
    
    def __init__(self, operator, storage, loop = -1):
        self.operator = operator
        self.atom = storage.load("h_loc_diagonalization", loop, bcast = False)
        self.rho = storage.load("density_matrix")

    def get_expectation_value(self):
        return trace_rho_op(self.rho, self.operator, self.atom)

    def get_expectation_value_statewise(self):
        evs = []
        for i in range(self.atom.full_hilbert_space_dim):
            state = np.zeros([self.atom.full_hilbert_space_dim])
            state[i] = 1
            evs.append(state.dot(act(self.operator, state, self.atom)))
        return evs
