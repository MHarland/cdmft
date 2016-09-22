from pytriqs.applications.impurity_solvers.cthyb import AtomDiag, trace_rho_op

class StaticObservable:
    
    def __init__(self, operator, storage):
        self.operator = operator
        self.atom = self.storage.load("h_loc_diagonalization", loop, bcast = False)
        self.rho = self.storage.load("density_matrix")

    def get_expectation_value(self, loop = -1):
        return trace_rho_op(self.rho, self.operator, self.atom)
