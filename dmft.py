from mpi4py import MPI
from time import time

from impuritysolver import ImpuritySolver
from greensfunctions import GImpurity
from storage import LoopStorage


class DMFT:

    def __init__(self, loopstorage, parameters, weiss_field, h_int, g_local, mu, self_energy):
        """
        mu and self-energy data initialize the first loop. g_local is required
        due to the self-consistency equation, that belongs to that class.
        """
        self.h_int = h_int
        self.p = parameters
        self.storage = loopstorage
        self.imp_solver = ImpuritySolver(*self.p.init_solver())
        self.g0 = weiss_field
        self.g_loc = g_local
        self.g_imp = GImpurity(**self.p.init_gf_iw())
        self.mu = mu
        self.se = self_energy

    def run_loops(self, n_loops, **parameters_dict):
        self.p.set(parameters_dict)
        for i in range(n_loops):
            loop_nr = self.storage.get_completed_loops()
            self.report("DMFT loop nr. "+str(loop_nr)+":")
            loop_time = time()
            self.mu = self.g_loc.set_mu(self.se, self.mu, self.p["fit_min_w"], self.p["fit_max_w"], self.p["filling"], self.p["dmu_max"])
            self.g0.calc_selfconsistency(self.g_loc, self.mu)
            self.prepare_impurity_run()
            self.imp_solver.run(self.g0, self.h_int, loop_nr, **self.p.run_solver())
            self.g_imp.set_gf(self.imp_solver.get_g_iw())
            self.process_impurity_results()
            results = {}
            results.update(self.imp_solver.get_results())
            results.update({"g_loc_iw": self.g_loc.gf,
                            "se_imp_iw": self.se.gf,
                            "g_imp_iw": self.g_imp.gf,
                            "density0": self.g_loc.total_density(),
                            "mu": self.mu,
                            "density": self.g_imp.total_density(),
                            "loop_time": loop_time - time()})
            self.storage.save_loop(results)
            self.report_variable(average_sign = results["average_sign"],
                                 density = results["density"])
            self.report("Loop done.")

    def process_impurity_results(self):
        """
        processes the impurity results on the level of the self-energy,
        updates g_imp accordingly
        """
        self.se.set_gf(self.imp_solver.get_se())
        self.se.mix(self.p["mix"])
        self.se.symmetrize(self.p["block_symmetries"])
        self.g_imp.calc_dyson(self.g0, self.se)

    def prepare_impurity_run(self):
        if self.p["make_g0_tau_real"]:
            self.g0.make_g_tau_real(self.p["n_tau"])
        self.se.prepare_mix()

    def report(self, text):
        comm = MPI.COMM_WORLD
        if comm.rank == 0:
            print text

    def report_variable(self, **variables):
        for key, val in variables.items():
            self.report(key+" = "+str(val))
