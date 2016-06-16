import itertools as itt

from impuritysolver import impurity_solver_parameters as isp
from greensfunctions import matsubara_greensfunction_parameters as mgfp

class Parameters:
    
    def __init__(self, parameter_dict):
        self.known = ["beta", "n_iw"]
        for act, prior in itt.product(["run", "init"], ["obligatory", "optional"]):
            self.known += isp[act][prior]
        self.obligatory = ["name", "beta"] + isp["init"]["obligatory"]
        self.current = {}
        #self.default = {"n_iw": 1025}
        self.impurity_solver = {"init": {}, "run": {}}
        self.matsubara_greensfunction = {}
        self.set(parameter_dict)

    def set(self, parameter_dict):
        for par, val in parameter_dict.items():
            if par in mgfp:
                self.matsubara_greensfunction[par] = val
            for act, prior in itt.product(["run", "init"], ["obligatory", "optional"]):
                if par in isp[act][prior]:
                    self.impurity_solver[act][par] = val
            if par in self.known:
                self.current[par] = parameter_dict.pop(par)
        if len(parameter_dict) > 0:
            raise UnkownParameters(parameter_dict)

    def __getitem__(self, parameter_name):
        return self.current[parameter_name]

    def __iter__(self):
        for name, value in self.current.items():
            yield name, value

    def check_for_missing(self):
        missing = []
        for name in self.obligatory:
            if not name in self.current.keys():
                missing.append(name)
        raise MissingParameters(missing)

    
class UnkownParameters(Exception):

    def __init__(self, parameter_dict):
        self.names = [key for key in parameter_dict.keys()]
        self.message = "Parameters " + str(self.names) + " not recognized"

        
class MissingParameters(Exception):
    
    def __init__(self, names):
        self.names = names
        self.message = "Parameters " + str(self.names) + " are missing"
