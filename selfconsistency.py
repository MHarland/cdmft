from mpi4py import MPI
from time import time

from schemes.generic import GLocalGeneric
from impuritysolver import ImpuritySolver


class Cycle:

    def __init__(self, loopstorage, parameters, h_int, g_local, weiss_field, self_energy, mu, **kwargs):
        """
        mu and self-energy data initialize the first loop. g_local is required
        due to the self-consistency equation, that belongs to that class.
        optional kwargs are global_moves and quantum_numbers
        """
        self.h_int = h_int
        p = self.p = parameters
        for kwargkey in kwargs.keys():
            if kwargkey in ['global_moves', 'quantum_numbers']:
                p[kwargkey] = kwargs[kwargkey]
        self.storage = loopstorage
        g0 = self.g0 = weiss_field
        self.imp_solver = ImpuritySolver(g0.beta, dict(g0.gf_struct), g0.n_iw, p['n_tau'], p['n_l'])
        self.g_loc = g_local
        self.g_imp = GLocalGeneric(gf_init = g_local)
        self.mu = mu
        self.se = self_energy

    def run(self, n_loops, **parameters_dict):
        """
        parameters are taken from initialization, but can also be updated using the optional argument
        """
        self.p.set(parameters_dict)
        for i in range(n_loops):
            loop_nr = self.storage.get_completed_loops()
            self.report("DMFT loop nr. "+str(loop_nr)+":")
            self.start_time = time()
            self.mu = self.g_loc.set(self.se, self.mu, self.p["filling"], self.p["dmu_max"])
            self.g0.calc_selfconsistency(self.g_loc, self.se, self.mu)
            self.prepare_impurity_run()
            self.imp_solver.run(self.g0, self.h_int, loop_nr, **self.p.run_solver())
            self.g_imp << self.imp_solver.get_g_iw()
            self.process_impurity_results()
            self.save()
            self.report("Loop done.")

    def save(self):
        results = {}
        results.update(self.imp_solver.get_results())
        results.update({"g_loc_iw": self.g_loc.get_as_BlockGf(),
                        "se_imp_iw": self.se.get_as_BlockGf(),
                        "g_imp_iw": self.g_imp.get_as_BlockGf(),
                        "g_weiss_iw": self.g0.get_as_BlockGf(),
                        "density0": self.g_loc.total_density(),
                        "mu": self.mu,
                        "density": self.g_imp.total_density(),
                        "loop_time": time() - self.start_time})
        self.storage.save_loop(results)
        self.report_variable(average_sign = results["average_sign"],
                             density = results["density"],
                             loop_time = results["loop_time"])

    def process_impurity_results(self):
        """
        processes the impurity results on the level of the self-energy,
        updates g_imp accordingly
        """
        self.se << self.imp_solver.get_se()
        self.se.mix(self.p["mix"])
        self.se.symmetrize(self.p["block_symmetries"])
        self.g_imp.calc_dyson(self.g0, self.se)

    def prepare_impurity_run(self):
        if self.p["make_g0_tau_real"]:
            self.g0.make_g_tau_real(self.p["n_tau"])
        self.se.prepare_mix()

    def report(self, text):
        comm = MPI.COMM_WORLD
        if comm.rank == 0 and 0 < self.p["verbosity"]:
            print text

    def report_variable(self, **variables):
        for key, val in variables.items():
            self.report(key+" = "+str(val))
