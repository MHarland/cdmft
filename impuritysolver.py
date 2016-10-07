from pytriqs.applications.impurity_solvers.cthyb import Solver
from pytriqs.gf.local import SemiCircular, LegendreToMatsubara, TailGf, BlockGf, GfImFreq, inverse
from pytriqs.random_generator import random_generator_names_list
from pytriqs.utility import mpi


class ImpuritySolver:
    
    def __init__(self, beta, gf_struct, n_iw, n_tau, n_l, *args, **kwargs):
        """
        parameters
        init:
        (required: beta, gf_struct)
        standard: n_iw = 1025, n_tau = 10001, n_l = 50
        run:
        required: h_int n_cycles
        optional: partition_method, quantum_numbers, length_cycle, n_warmup_cycles, random_seed, random_name, max_time, verbosity, move_shift, move_double, use_trace_estimator, measure_g_tau, measure_g_l, measure_pert_order, measure_density_matrix, use_norm_as_weight, performance_analysis, proposal_prob, imag_threshold
        """
        self.gf_struct = gf_struct
        self.beta = beta
        self.n_iw = n_iw
        self.run_parameters = {}
        self.cthyb = Solver(beta, dict(gf_struct), n_iw, n_tau, n_l, *args, **kwargs)

    def get_g_iw(self, by_tau = False, by_legendre = True):
        assert by_tau ^ by_legendre, "G_iw can only be set by one G of the solver, since it is used for the next dmft loop"
        if by_tau:
            return self._get_g_iw_by_tau()
        elif by_legendre:
            return self._get_g_iw_by_legendre()

    def _init_new_giw(self):
        return BlockGf(name_list = [b[0] for b in self.gf_struct],
                       block_list = [GfImFreq(n_points = self.n_iw, beta = self.beta,
                                              indices = b[1]) for b in self.gf_struct]
                       )

    def _get_g_iw_by_tau(self):
        g_iw = self._init_new_giw()
        if self.run_parameters["perform_post_proc"]:
            g_iw << inverse(inverse(self.cthyb.G0_iw) - self.cthyb.Sigma_iw)
        else:
            g_iw << self.cthyb.G_iw
        return g_iw

    def _get_g_iw_by_legendre(self):
        g_iw = self._init_new_giw()
        for s, b in self.cthyb.G_l:
            g_iw[s] << LegendreToMatsubara(b)
        return g_iw

    def get_se(self, by_tau = False, by_legendre = True):
        se = self._init_new_giw()
        if by_tau or not self.run_parameters["measure_g_l"]:
            for s, b in self.cthyb.Sigma_iw:
                se[s] << b
        else:
            assert by_legendre and self.run_parameters["measure_g_l"], "Need either g_legendre or g_tau to set sigma_iw"
            g_iw = self._get_g_iw_by_legendre()
            se << inverse(self.cthyb.G0_iw) - inverse(g_iw)
        return se

    def _get_internal_parameters(self, loop_nr):
        par = {}
        rnames = random_generator_names_list()
        n_r = len(rnames)
        par.update({'random_name': rnames[int((loop_nr + mpi.rank) % n_r)]})
        return par

    def run(self, weiss_field, hamiltonian, loop_nr, **run_parameters):
        self.cthyb.G0_iw << weiss_field.gf
        self.run_parameters["h_int"] = hamiltonian
        self.run_parameters.update(run_parameters)
        self.run_parameters.update(self._get_internal_parameters(loop_nr))
        self.cthyb.solve(**self.run_parameters)

    def get_results(self):
        params = self.cthyb.last_solve_parameters
        results = {"last_solve_parameters": self.cthyb.last_solve_parameters,
                   "g0_iw": self.cthyb.G0_iw,
                   "delta_tau": self.cthyb.Delta_tau,
                   "atomic_gf": self.cthyb.atomic_gf,
                   "h_loc_diagonalization": self.cthyb.h_loc_diagonalization,
                   "average_sign": self.cthyb.average_sign,
                   "solve_status": self.cthyb.solve_status}
        if params["measure_g_tau"]:
            results.update({"g_tau": self.cthyb.G_tau})
            if "perform_post_proc" in self.run_parameters.keys():
                if self.run_parameters["perform_post_proc"]:
                    results.update({"sigma_sol_iw": self.cthyb.Sigma_iw})
                    results.update({"g_sol_iw": self.cthyb.G_iw})
        if params["measure_g_l"]:
            results.update({"g_sol_l": self.cthyb.G_l})
        if params["measure_density_matrix"]:
            results.update({"density_matrix": self.cthyb.density_matrix})
        if params["measure_pert_order"]:
            results.update({"perturbation_order_total": self.cthyb.perturbation_order_total})
            results.update({"perturbation_order": self.cthyb.perturbation_order})
        if params["performance_analysis"]:
            results.update({"performance_analysis": self.cthyb.performance_analysis})
        return results
