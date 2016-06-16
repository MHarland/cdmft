from pytriqs.applications.impurity_solvers.cthyb import Solver
from pytriqs.gf.local import SemiCircular

class ImpuritySolver:
    
    def __init__(self, beta, gf_struct, *args, **kwargs):
        """TRIQS standard: n_iw = 1025, n_tau = 10001, n_l = 50"""
        self.run_parameters = {}
        self.cthyb = Solver(beta, gf_struct, *args, **kwargs)

    def prepare(self, weiss_field, hamiltonian, **run_parameters):
        self.cthyb.G0_iw << weiss_field
        self.run_parameters["h_int"] = hamiltonian
        self.run_parameters.update(run_parameters)

    def run(self):
        self.cthyb.solve(**self.run_parameters)

    def list_results(self):
        params = self.cthyb.last_solve_parameters
        results_list = [self.cthyb.last_solve_parameters,
                        self.cthyb.G0_iw,
                        self.cthyb.Delta_tau,
                        self.cthyb.atomic_gf,
                        self.cthyb.h_loc_diagonalization,
                        self.cthyb.average_sign,
                        self.cthyb.solve_status]
        if params["measure_g_tau"]:
            results_list.append(self.cthyb.G_tau)
        if params["measure_g_l"]:
            results_list.append(self.cthyb.G_l)
        if params["measure_density_matrix"]:
            results_list.append(self.cthyb.density_matrix)
        if params["measure_pert_order"]:
            results_list.append(self.cthyb.perturbation_order_total)
            results_list.append(self.cthyb.perturbation_order)
        if params["performance_analysis"]:
            results_list.append(self.cthyb.performance_analysis)
        return results_list
    
impurity_solver_parameters_init = {"obligatory": ["beta", "gf_struct"],
                                   "optional": ["n_iw", "n_tau", "n_l"]}
impurity_solver_parameters_run = {"obligatory": ["h_int", "n_cycles"],
                                  "optional": ["partition_method", "quantum_numbers", "length_cycle", "n_warmup_cycles", "random_seed", "random_name", "max_time", "verbosity", "move_shift", "move_double", "use_trace_estimator", "measure_g_tau", "measure_g_l", "measure_pert_order", "measure_density_matrix", "use_norm_as_weight", "performance_analysis", "proposal_prob", "imag_threshold"]}
impurity_solver_parameters = {"init": impurity_solver_parameters_init,
                              "run": impurity_solver_parameters_run}
