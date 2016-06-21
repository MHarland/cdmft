import itertools as itt


class DMFTParameters:
    """
    always needs gf_struct and beta
    untouched parameters: random_seed, fit_known_moments
    treated in system: h_int, t, t_loc, mu, u, initial_guess, gf_struct, beta
    """
    def __init__(self, parameter_dict = {}):
        self.solver_run = ["n_cycles", "partition_method", "quantum_numbers", "length_cycle", "n_warmup_cycles", "random_name", "max_time", "verbosity", "move_shift", "move_double", "use_trace_estimator", "measure_g_tau", "measure_g_l", "measure_pert_order", "measure_density_matrix", "use_norm_as_weight", "performance_analysis", "proposal_prob", "imag_threshold", "perform_post_proc", "perform_tail_fit", "fit_min_n", "fit_max_n", "fit_min_w", "fit_max_w", "fit_max_moment"]
        all_parameternames = ["beta", "n_iw", "n_tau", "n_l", "gf_struct"] + self.solver_run
        self.current = dict([(name, None) for name in all_parameternames])
        self.set(parameter_dict)

    def run_solver(self):
        rs_dict = {}
        for par, val in self.current.items():
            if self._check_run_parameter(par, val):
                rs_dict[par] = val
        return rs_dict

    def _check_run_parameter(self, par, val):
        if not self.current["perform_tail_fit"]:
            if par in ["fit_min_n", "fit_max_n", "fit_min_w", "fit_max_w", "fit_max_moment"]:
                return False
        if not self.current["perform_post_proc"]:
            if par in ["perform_tail_fit", "fit_min_n", "fit_max_n", "fit_min_w", "fit_max_w", "fit_max_moment"]:
                return False
        if not (par in self.solver_run):
            return False
        return True

    def init_solver(self):
        return [self.current["beta"], self.current["gf_struct"], self.current["n_iw"], self.current["n_tau"], self.current["n_l"]]

    def init_gf_iw(self):
        name_list = [block[0] for block in self.current["gf_struct"]]
        block_states = [block[1] for block in self.current["gf_struct"]]
        beta = self.current["beta"]
        n_iw = self.current["n_iw"]
        return [name_list, block_states, beta, n_iw]

    def assert_setup_complete(self):
        """
        problematic, since None can occur as input for e.g. fit_max_n
        """
        missing = []
        for par, val in self.current.items():
            if val is None:
                missing.append(par)
        if len(missing) > 0:
            raise MissingParameters(missing)

    def set(self, parameter_dict):
        for par, val in parameter_dict.items():
            if par in self.current.keys():
                self.current[par] = parameter_dict.pop(par)
        if len(parameter_dict) > 0:
            raise UnkownParameters(parameter_dict)

    def __call__(self, parameters = {}):
        self.current.update(parameters)
        return self

    def __getitem__(self, parameter_name):
        return self.current[parameter_name]

    def __setitem__(self, parameter_name, parameter_value):
        self.current[parameter_name] = parameter_value

    def __iter__(self):
        for name, value in self.current.items():
            yield name, value

    def missing(self):
        missing = []
        for par, val in self.current.items():
            if val is None:
                missing.append(par)
        return missing

    
class UnkownParameters(Exception):

    def __init__(self, parameter_dict, verbosity = 0):
        self.names = [key for key in parameter_dict.keys()]
        self.message = "Parameters " + str(self.names) + " not recognized"
        if verbosity > 0:
            print self.message

        
class MissingParameters(Exception):
    
    def __init__(self, names, verbosity = 0):
        self.names = names
        self.message = "Parameters " + str(self.names) + " are missing"
        if verbosity > 0:
            print self.message


class DefaultDMFTParameters(DMFTParameters):

    def __init__(self, parameter_dict = {}):
        default = {"n_iw": 1025,
                   "n_tau": 10001,
                   "n_l": 30,
                   # solver:
                   "n_cycles": 1000,
                   "partition_method": "autopartition",
                   "quantum_numbers": [],
                   "length_cycle": 10,
                   "n_warmup_cycles": 500,
                   "random_name": "",
                   "max_time": -1,
                   "verbosity": 0,
                   "move_shift": True,
                   "move_double": True,
                   "use_trace_estimator": False,
                   "measure_g_tau": True,
                   "measure_g_l": False,
                   "measure_pert_order": False,
                   "measure_density_matrix": False,
                   "use_norm_as_weight": False,
                   "performance_analysis": False,
                   "proposal_prob": {},
                   "imag_threshold": 1.e-15,
                   # uses solver's fitting to the self-energy
                   "perform_post_proc": True,
                   "perform_tail_fit": True,
                   "fit_min_n": 800,
                   "fit_max_n": 1025,
                   "fit_min_w": None,
                   "fit_max_w": None,
                   "fit_max_moment": 3}
        default.update(parameter_dict)
        DMFTParameters.__init__(self, default)
