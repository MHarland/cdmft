from pytriqs.utility import mpi


class DMuMaxSqueezer:
    def __init__(self, gloc, gimp, squeeze = False, desired_filling = None, factor = .5, verbosity = 0, par = {}):
        self.gloc = gloc
        self.gimp = gimp
        self.squeeze = squeeze
        self.f = desired_filling
        self.factor = factor
        self.verbosity = verbosity
        for key, val in par.items():
            if key == 'squeeze_dmu_max': self.squeeze = val
            if key == 'filling': self.f = val
            if key == 'dmu_max_squeeze_factor': self.factor = val
            if key == 'verbosity': self.verbosity = val
        if self.squeeze: assert self.f, 'squeeze needs filling'

    def __call__(self, dmu_max):
        if self.squeeze:
            nloc = self.gloc.total_density().real
            nimp = self.gimp.total_density().real
            if (self.f - nloc)/(self.f - nimp) < 0:
                dmu_max *= self.factor
                if mpi.is_master_node() and self.squeeze:
                    print 'setting dmu_max to', dmu_max
        return dmu_max
