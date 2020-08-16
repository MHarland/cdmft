import numpy as np
import itertools as itt
from pytriqs.gf import BlockGf, GfImTime, InverseFourier

from cdmft.evaluation.common import Evaluation as CommonEvaluation


class Evaluation(CommonEvaluation):
    def __init__(self, archive):
        self.archive = archive
        self.n_loops = self.archive.get_completed_loops()

    def get_scorder(self, loop=-1):
        g = self.get_g_imp_tau(loop)
        scos = np.array([-g["X"].data[-1, 0, 1].real,
                         -g["X"].data[-1, 1, 0].real,
                         -g["Y"].data[-1, 0, 1].real,
                         -g["Y"].data[-1, 1, 0].real])
        return np.mean(abs(scos)), np.std(abs(scos))

    def get_scorderset_loop(self):
        scoset_loop = np.empty([self.n_loops, 4])
        for i in range(self.n_loops):
            g = self.get_g_imp_tau()
            scoset_loop[i, 0] = -g["X"].data[-1, 0, 1].real
            scoset_loop[i, 1] = -g["X"].data[-1, 1, 0].real
            scoset_loop[i, 2] = -g["Y"].data[-1, 0, 1].real
            scoset_loop[i, 3] = -g["Y"].data[-1, 1, 0].real
        return scoset_loop

    def get_scorder_loop(self):
        scoset_loop = self.get_scorderset_loop()
        for scoset, j in itt.product(scoset_loop, range(4)):
            scoset[j] = abs(scoset[j])
        return np.array([[i, np.mean(scoset_loop[i, :]), np.std(scoset_loop[i, :], ddof=1)] for i in range(self.n_loops)])

    def get_g_imp_tau(self, loop=-1):
        giw = self.archive.load("g_imp_iw", loop)
        g_imp_tau = BlockGf(name_block_generator=[(s, GfImTime(
            beta=giw.mesh.beta, n_points=10001, indices=[i for i in b.indices])) for s, b in giw], make_copies=False)
        for s, b in giw:
            g_imp_tau[s] = InverseFourier(b)
        return g_imp_tau
