import numpy as np, itertools as itt

from bethe.storage import LoopStorage
from bethe.evaluation.generic import Evaluation as GenericEvaluation


class Evaluation(GenericEvaluation):
    def __init__(self, archive):
        self.archive = LoopStorage(archive)
        self.n_loops = self.archive.get_completed_loops()

    def get_scorder(self, loop = -1):
        g = self.archive.load("g_tau", loop)
        scos = np.array([-g["X"].data[-1,0,1].real,
                        -g["X"].data[-1,1,0].real,
                        -g["Y"].data[-1,0,1].real,
                        -g["Y"].data[-1,1,0].real])
        return np.mean(abs(scos)), np.std(abs(scos))

    def get_scorderset_loop(self):
        scoset_loop = np.empty([self.n_loops, 4])
        for i in range(self.n_loops):
            g = self.archive.load("g_tau", i)
            scoset_loop[i, 0] = -g["X"].data[-1,0,1].real
            scoset_loop[i, 1] = -g["X"].data[-1,1,0].real
            scoset_loop[i, 2] = -g["Y"].data[-1,0,1].real
            scoset_loop[i, 3] = -g["Y"].data[-1,1,0].real
        return scoset_loop
        
    def get_scorder_loop(self):
        scoset_loop = self.get_scorderset_loop()
        for scoset, j in itt.product(scoset_loop, range(4)):
            scoset[j] = abs(scoset[j])
        return np.array([[i, np.mean(scoset_loop[i,:]), np.std(scoset_loop[i,:], ddof=1)] for i in range(self.n_loops)])
