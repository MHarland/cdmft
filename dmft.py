from impuritysolver import ImpuritySolver
from weissfield import WeissField
from glocal import GLocal
from storage import LoopStorage


class DMFT:

    def __init__(self, parameters, model, archive_name):
        self.model = model
        self.par = parameters({"beta": self.model.beta, "gf_struct": self.model.gf_struct})
        self.storage = LoopStorage(archive_name)
        self.impurity_solver = ImpuritySolver(*self.par.init_solver())
        self.g0 = WeissField(*(self.par.init_gf_iw() + [self.model.t, self.model.t_loc]))
        if self.storage.provide_last_g_loc() is None:
            self.g_loc = self.model.initial_guess.copy()
        else:
            self.g_loc = self.storage.provide_last_g_loc()
        self._g_loc_old = self.g_loc.copy()

    def run_loops(self, n_loops, **parameters_dict):
        self.par.set(parameters_dict)
        for i in range(n_loops):
            self.g0.calc_selfconsistency(self.g_loc, self.model.mu)
            self.impurity_solver.run(self.g0.gf, self.model.h_int, **self.par.run_solver())
            self.g_loc = self.impurity_solver.get_g_iw()
            self.process_impurity_results()
            self.storage.save_loop(self.impurity_solver.get_results(),
                                   {"g_loc_iw": self.g_loc})

    def process_impurity_results(self):
        self.g_loc = self.mix(self.g_loc, self.par["mix"])

    def mix(self, g, mix):
        g << mix * self.g_loc + (1 - mix) * self._g_loc_old
        self._g_loc_old << g
        return g
