from impuritysolver import ImpuritySolver
from weissfield import WeissField
from glocal import GLocal


class DMFT:

    def __init__(self, parameters, hamiltonian):
        self.params = Parameters(parameters)
        self.params.check_for_missing()
        #self.storage = Storage()
        
        self.impuity_solver = ImpuritySolver(**self.params.impurity_solver["init"])
        self.g0 = WeissField(**self.params.matsubara_greensfunction)
        self.g_loc = GLocal(**self.params.matsubara_greensfunction)
        self.mu = 
        self.hamiltonian = hamiltonian
        

    def run_loops(self, n_loops):
        self.g0.calc_selfconsistency(self.g_loc, self.mu)
