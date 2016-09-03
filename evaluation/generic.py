import numpy as np

from bethe.storage import LoopStorage


class Evaluation:
    
    def __init__(self, archive):
        self.archive = LoopStorage(archive)
        self.n_loops = self.archive.disk["dmft_results"]["n_dmft_loops"]

    def get_sign_loop(self):
        signs = np.empty([self.n_loops]*2)
        for i in range(self.n_loops):
            signs[i, 0] = i
            signs[i, 1] = self.archive.load("average_sign", i)
        return signs

    def get_density(self, loop = -1):
        return self.archive.load("density", loop)
