import numpy as np, itertools as itt
from pytriqs.gf.local import BlockGf, GfImFreq # replace by MatsubaraGreensFunction todo
from periodization.selfenergyperiodization import LatticeSelfenergy, LatticeGreensfunction

from ..greensfunctions import MatsubaraGreensFunction
from common import GLocalCommon, SelfEnergyCommon, WeissFieldCommon


class GLocal(GLocalCommon):
    """
    In fact it is G_cluster constructed from G_lattice, conceptual problem todo
    impurity_transformation is needed to backtransform into site-space before periodization
    """
    def __init__(self, glat_orb_struct, gcluster_orb_struct, r, weights_r, hopping_r, nk, imp_to_lat_r, lat_r_to_cluster, impurity_transformation, *args, **kwargs):
        GLocalCommon.__init__(self, *args, **kwargs)
        self.transf = impurity_transformation
        self.se_lat_initdict = {'blocknames': glat_orb_struct.keys(),
                                'blockindices': glat_orb_struct.values(),
                                'r': r,
                                'hopping_r': hopping_r,
                                'nk': nk,
                                'weights_r': weights_r}
        self.imp_to_lat_r = imp_to_lat_r
        self.lat_r_to_cluster = lat_r_to_cluster
        self.se_lat_tmp = BlockGf(name_block_generator = [[bn, GfImFreq(indices = b, mesh = self.mesh)] for bn, b in glat_orb_struct.items()])
        self.g_cluster = BlockGf(name_block_generator = [[bn, GfImFreq(indices = b, mesh = self.mesh)] for bn, b in gcluster_orb_struct.items()])
        self.g_lat = None
        self.se_lat = None
        
    def set(self, se_imp, mu):
        """
        sets GLocal using calculate(self, mu, selfenergy, w1, w2, n_mom), uses either filling or mu
        mu can be either of blockmatrix-type or scalar
        in the case of ccdmft the selfenergy gets changed during the calculation of gloc
        """
        if not(self.filling is None):
            assert False, 'todo'
            #mu = self.find_and_set_mu(self.filling, se_lat, mu, self.dmu_max)
        if self.transf is not None:
            se_imp = self.transf.backtransform_g(se_imp)
        self.se_lat_initdict['sigma_r'] = self.calc_se_r(se_imp)
        self.se_lat = LatticeSelfenergy(**self.se_lat_initdict)
        self.se_lat.periodize()
        self.g_lat = LatticeGreensfunction(self.se_lat, mu)
        for i_lat, i_cluster in self.lat_r_to_cluster.items():
            c0, c1, c2 = i_cluster
            l0, l1, l2, l3 = i_lat
            self.g_cluster[c0][c1, c2] << self.g_lat[l0][l1][l2, l3]
        if self.transf is not None:
            self << self.transf.transform_g(self.g_cluster)
        else:
            self << self.g_cluster
        return mu

    def calc_se_r(self, se_imp):
        se_r = []
        for imp_to_lat in self.imp_to_lat_r:
            self.se_lat_tmp.zero()
            for i_imp, i_lat in imp_to_lat.items():
                i0, i1, i2 = i_lat
                j0, j1, j2 = i_imp
                self.se_lat_tmp[i0][i1, i2] << se_imp[j0][j1, j2]
            se_r.append(self.se_lat_tmp.copy())
        return se_r

    def calc_g_cluster(self, g_lat):
        for i_lat, i_cluster in self.lat_r_to_cluster.items():
            i0, i1, i2 = i_cluster
            jr0, jr1, j0, j1, j2 = i_lat
            self.g_cluster[i0][i1, i2] << self.g_lat[jr0, jr1][j0][j1, j2]


class SelfEnergy(SelfEnergyCommon):
    pass


class WeissField(WeissFieldCommon):
    pass

