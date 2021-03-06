import itertools as itt


class DMFTParameters:
    """
    Collects all dmft parameters. Furthermore it is used by the dmft class to (partially) initialize objects, i.e. the solver and greens functions.
    Untouched parameters: random_seed, fit_known_moments.
    """

    def __init__(self, parameter_dict={}, **kwargs):
        self.solver_run = ["n_cycles", "partition_method", "quantum_numbers", "length_cycle", "n_warmup_cycles", "random_name", "max_time", "verbosity", "move_shift", "move_double", "use_trace_estimator", "measure_G_tau", "measure_G_l", "measure_pert_order",
                           "measure_density_matrix", "use_norm_as_weight", "performance_analysis", "proposal_prob", "imag_threshold", "perform_post_proc", "perform_tail_fit", "fit_min_n", "fit_max_n", "fit_min_w", "fit_max_w", "fit_max_moment", "move_global", "move_global_prob"]
        all_parameternames = ["beta", "n_iw", "n_tau", "n_l", "mix", "make_g0_tau_real", "filling",
                              "block_symmetries", "dmu_max", "squeeze_dmu_max", "dmu_max_squeeze_factor"] + self.solver_run
        measure_g2_parameters = ["measure_g2_inu", "measure_g2_legendre", "measure_g2_pp", "measure_g2_ph",
                                 "measure_g2_block_order", "measure_g2_n_iw", "measure_g2_n_inu", "measure_g2_n_l", "measure_g_pp_tau", "measure_g2_blocks"]
        self.measure_g2_parameters = measure_g2_parameters
        self.current = dict([(name, None) for name in all_parameternames])
        parameter_dict.update(kwargs)
        self.set(parameter_dict)

    def items(self):
        return self.current.items()

    def run_solver(self):
        """
        Returns all valid parameters for the solver.run().
        """
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
        solver_parameters = self.solver_run
        if 'measure_g2_inu' in self.current.keys() or 'measure_g2_legendre' in self.current.keys():
            if self.current['measure_g2_inu'] or self.current['measure_g2_legendre']:
                solver_parameters += self.measure_g2_parameters
            if not (par in solver_parameters):
                return False
        if not par in solver_parameters:
            return False
        return True

    def init_g_iw(self):  # TODO deprecated?
        blocknames = [block[0]
                      for block in self.current["gf_struct"]]  # gf_struct removed!
        blocksizes = [len(block[1]) for block in self.current["gf_struct"]]
        beta = self.current["beta"]
        n_iw = self.current["n_iw"]
        return {"blocknames": blocknames, "blocksizes": blocksizes, "beta": beta, "n_iw": n_iw}

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
            for par in parameter_dict.keys():
                if not (par in self.measure_g2_parameters):
                    print par
                    raise UnkownParameters(parameter_dict)
                else:
                    self.current[par] = parameter_dict.pop(par)

    def __call__(self, parameters={}):
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

    def __init__(self, parameter_dict, verbosity=0):
        self.names = [key for key in parameter_dict.keys()]
        self.message = "Parameters " + str(self.names) + " not recognized"
        if verbosity > 0:
            print self.message


class MissingParameters(Exception):

    def __init__(self, names, verbosity=0):
        self.names = names
        self.message = "Parameters " + str(self.names) + " are missing"
        if verbosity > 0:
            print self.message


class DefaultDMFTParameters(DMFTParameters):

    def __init__(self, parameter_dict={}):
        default = {"n_iw": 1025,
                   "n_tau": 10001,
                   "n_l": 30,
                   "mix": 1,
                   "make_g0_tau_real": False,
                   "filling": None,
                   "block_symmetries": [],
                   "dmu_max": 10,
                   "squeeze_dmu_max": False,
                   "dmu_max_squeeze_factor": .5,
                   # solver:
                   "n_cycles": 10**5,
                   "partition_method": "autopartition",  # "quantum_numbers"
                   "quantum_numbers": [],
                   "length_cycle": 50,
                   "n_warmup_cycles": 5*10**3,
                   "random_name": "",
                   "max_time": -1,
                   "verbosity": 2,
                   "move_shift": True,
                   "move_double": True,
                   "use_trace_estimator": False,
                   "measure_G_tau": True,
                   "measure_G_l": True,
                   "measure_pert_order": False,
                   "measure_density_matrix": False,
                   # "measure_g_pp_tau": False,
                   "use_norm_as_weight": False,
                   "performance_analysis": False,
                   "proposal_prob": {},
                   "imag_threshold": 1.e-10,
                   "move_global": {},
                   "move_global_prob": 0.05,
                   # uses solver's fitting to the self-energy
                   "perform_post_proc": True,
                   "perform_tail_fit": True,
                   "fit_min_n": 800,
                   "fit_max_n": 1025,
                   "fit_min_w": None,
                   "fit_max_w": None,
                   "fit_max_moment": 3}
        """
                   "measure_g2_inu": False,
                   "measure_g2_legendre": False,
                   "measure_g2_pp": False,
                   "measure_g2_ph": False,
                   "measure_g2_block_order": 'AABB',
                   #"measure_g2_blocks": measure all blocks,
                   "measure_g2_n_iw": 30,
                   "measure_g2_n_inu": 30,
                   "measure_g2_n_l": 20}
        """
        default.update(parameter_dict)
        DMFTParameters.__init__(self, default)


class MeasureG2DMFTParameters(DMFTParameters):

    def __init__(self, parameter_dict={}):
        default = {"n_iw": 1025,
                   "n_tau": 10001,
                   "n_l": 30,
                   "mix": 1,
                   "make_g0_tau_real": False,
                   "filling": None,
                   "block_symmetries": [],
                   "dmu_max": 10,
                   # solver:
                   "n_cycles": 10**5,
                   "partition_method": "autopartition",  # "quantum_numbers"
                   "quantum_numbers": [],
                   "length_cycle": 50,
                   "n_warmup_cycles": 5*10**3,
                   "random_name": "",
                   "max_time": -1,
                   "verbosity": 2,
                   "move_shift": True,
                   "move_double": True,
                   "use_trace_estimator": False,
                   "measure_G_tau": True,
                   "measure_G_l": False,
                   "measure_pert_order": False,
                   "measure_density_matrix": False,
                   # "measure_g_pp_tau": False,
                   "use_norm_as_weight": False,
                   "performance_analysis": False,
                   "proposal_prob": {},
                   "imag_threshold": 1.e-10,
                   "move_global": {},
                   "move_global_prob": 0.05,
                   # uses solver's fitting to the self-energy
                   "perform_post_proc": True,
                   "perform_tail_fit": True,
                   "fit_min_n": 800,
                   "fit_max_n": 1025,
                   "fit_min_w": None,
                   "fit_max_w": None,
                   "fit_max_moment": 3,
                   "measure_g2_inu": False,
                   "measure_g2_legendre": False,
                   "measure_g2_pp": False,
                   "measure_g2_ph": False,
                   "measure_g2_block_order": 'AABB',
                   # "measure_g2_blocks": measure all blocks,
                   "measure_g2_n_iw": 30,
                   "measure_g2_n_inu": 30,
                   "measure_g2_n_l": 20}
        default.update(parameter_dict)
        DMFTParameters.__init__(self, default)


class TestDMFTParameters(DMFTParameters):

    def __init__(self, parameter_dict={}, **kwargs):
        default = {"n_iw": 1025,
                   "n_tau": 10001,
                   "n_l": 30,
                   "mix": 1,
                   "make_g0_tau_real": False,
                   "filling": None,
                   "block_symmetries": [],
                   "dmu_max": 10,
                   # solver:
                   "n_cycles": 2*10**5,
                   "partition_method": "autopartition",
                   "quantum_numbers": [],
                   "length_cycle": 15,
                   "n_warmup_cycles": 5*10**3,
                   "random_name": "",
                   "max_time": -1,
                   "verbosity": 0,
                   "move_shift": True,
                   "move_double": False,
                   "use_trace_estimator": False,
                   "measure_G_tau": False,
                   "measure_G_l": True,
                   "measure_pert_order": False,
                   "measure_density_matrix": False,
                   # "measure_g_pp_tau": False,
                   "use_norm_as_weight": False,
                   "performance_analysis": False,
                   "proposal_prob": {},
                   "imag_threshold": 1,  # 1.e-10,
                   "move_global": {},
                   "move_global_prob": 0.05,
                   # uses solver's fitting to the self-energy
                   "perform_post_proc": True,
                   "perform_tail_fit": True,
                   "fit_min_n": 800,
                   "fit_max_n": 1025,
                   "fit_min_w": None,
                   "fit_max_w": None,
                   "fit_max_moment": 3}
        """
                   "measure_g2_inu": False,
                   "measure_g2_legendre": False,
                   "measure_g2_pp": False,
                   "measure_g2_ph": False,
                   "measure_g2_block_order": 'AABB',
                   #"measure_g2_blocks": measure all blocks,
                   "measure_g2_n_iw": 30,
                   "measure_g2_n_inu": 30,
                   "measure_g2_n_l": 20}
        """
        default.update(parameter_dict)
        default.update(kwargs)
        DMFTParameters.__init__(self, default)
