from impuritysolver import ImpuritySolver
from weissfield import WeissField
from glocal import GLocal
from storage import LoopStorage
from greensfunctions import MatsubaraGreensFunction


class DMFT:

    def __init__(self, parameters, model, archive_name, weiss_field = None, g_loc = None):
        self.model = model
        self.par = parameters
        self.par.set_by_model(model)
        self.storage = LoopStorage(archive_name)
        self.impurity_solver = ImpuritySolver(*self.par.init_solver())
        self.g0 = WeissField(**self.par.init_gf_iw())
        self.g_loc = GLocal(**self.par.init_gf_iw())
        self.se = MatsubaraGreensFunction(**self.par.init_gf_iw())
        self.g_loc.set_gf(g_loc, self.storage.provide("g_loc_iw"), self.model.initial_guess)
        self.g0.set_gf(weiss_field, self.storage.provide("g0_iw"))
        self.mu = self.storage.provide("mu") if self.storage.provide("mu") else self.model.mu

    def run_loops(self, n_loops, **parameters_dict):
        self.par.set(parameters_dict)
        for i in range(n_loops):
            if self.par["filling"]:
                self.mu = self.g_loc.find_and_set_mu(self.par["filling"], self.se)
            self.g0.calc_selfconsistency(self.g_loc, self.mu)
            self.prepare_impurity_run()
            self.impurity_solver.run(self.g0, self.model.h_int, **self.par.run_solver())
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
