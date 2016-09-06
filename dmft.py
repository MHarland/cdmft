from impuritysolver import ImpuritySolver
from weissfield import WeissField
from glocal import GLocal
from storage import LoopStorage
from greensfunctions import MatsubaraGreensFunction


class DMFT:

    def __init__(self, loopstorage, parameters, weiss_field, h_int, g_local, mu, self_energy):
        """
        Parameters, h_int, LoopStorage, weiss_field, glocal
        """
        self.h_int = h_int
        self.par = parameters
        self.storage = loopstorage
        self.impurity_solver = ImpuritySolver(*self.par.init_solver())
        self.g0 = weiss_field
        self.g_loc = g_local
        self.mu = mu
        self.se = self_energy

    def run_loops(self, n_loops, **parameters_dict):
        self.par.set(parameters_dict)
        for i in range(n_loops):
            if self.par["filling"]:
                self.mu = dict_to_number(self.mu)
                self.mu = self.g_loc.find_and_set_mu(self.par["filling"], self.se, self.mu, self.par["dmu_max"])
            self.g0.calc_selfconsistency(self.g_loc, self.mu)
            self.prepare_impurity_run()
            self.impurity_solver.run(self.g0, self.h_int, **self.par.run_solver())
            self.g_loc.set_gf(self.impurity_solver.get_g_iw())
            self.process_impurity_results()
            self.storage.save_loop(self.impurity_solver.get_results(),
                                   {"g_loc_iw": self.g_loc.gf,
                                    "density0": self.g_loc.last_found_density,
                                    "mu": self.mu,
                                    "density": self.g_loc.total_density()})

    def process_impurity_results(self):
        if self.par["perform_post_proc"]:
            self.se.set_gf(self.impurity_solver.get_se())
        self.se.mix(self.par["mix"])
        self.se.symmetrize(self.par["block_symmetries"])
        self.g_loc.calc_dyson(self.g0, self.se)

    def prepare_impurity_run(self):
        if self.par["make_g0_tau_real"]:
            self.g0.make_g_tau_real(self.par["n_tau"])

def dict_to_number(mu):
    if isinstance(mu, dict):
        for key, val in mu.items():
            return val[0, 0]
    return mu
