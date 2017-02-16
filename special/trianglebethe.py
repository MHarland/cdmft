import numpy as np, itertools as itt

from bethe.schemes.bethe import SelfEnergy


class SelfEnergyForSymmetrizationInTransformedBasis(SelfEnergy):
    """
    old_struct would be spin site basis and transf_mat in basis of old_struct
    symmetrizes spins and sites
    """
    def __init__(self, matrix_transformation = None, site_symmetry_classes = None, *args, **kwargs):
        SelfEnergy.__init__(self, *args, **kwargs)
        self.transform = matrix_transformation
        self.symmetry_classes = site_symmetry_classes

    def symmetrize(self, whatever):
        g_site = self.transform.backtransform_g(self)
        g_tmp = g_site.copy()
        g_tmp.zero()
        for s, b in g_site:
            for sym_class in self.symmetry_classes:
                norm = len(sym_class)
                for i1, i2 in itt.product(sym_class, sym_class):
                    g_tmp[s][i1[0], i1[1]] += g_site[s][i2[0], i2[1]] / norm
        spins = [i for i in g_site.indices]
        for s1, s2 in itt.product(spins, spins):
            g_site[s1] << .5 * (g_tmp[s2] + g_tmp[s2])
        self << self.transform.transform_g(g_site)
