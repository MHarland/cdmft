import numpy as np
import itertools as itt
from pytriqs.gf import inverse, iOmega_n, BlockGf, GfImFreq  # todo use mats gf
from pytriqs.lattice.tight_binding import TBLattice
from pytriqs.utility import mpi

from periodization.greensfunctionperiodization import LatticeSelfenergy, LatticeGreensfunction

from common import GLocalCommon, SelfEnergyCommon, WeissFieldCommon
from ..gfoperations import double_dot_product_ggg


class GLocal(GLocalCommon):
    """
    In fact it is G_cluster constructed from G_lattice, conceptual problem todo
    impurity_transformation is needed to backtransform into site-space before periodization
    gbar is glattice with one site/cluster removed
    lambd is the hybridization
    glocal = u g_cluster udag
    """

    def __init__(self, glat_orb_struct, gcluster_orb_struct, r, weights_r, hopping_r, nk, imp_to_lat_r, lat_r_to_cluster, impurity_transformation, r_cavity, r_cluster, *args, **kwargs):
        GLocalCommon.__init__(self, *args, **kwargs)
        self.transf = impurity_transformation
        self.g_lat_initdict = {'blocknames': glat_orb_struct.keys(),
                               'blockindices': glat_orb_struct.values(),
                               'r': r,
                               'hopping_r': hopping_r,
                               'nk': nk,
                               'weights_r': weights_r}
        self.imp_to_lat_r = imp_to_lat_r
        self.lat_r_to_cluster = lat_r_to_cluster
        self.g_lat_tmp = BlockGf(name_block_generator=[[bn, GfImFreq(
            indices=b, mesh=self.mesh)] for bn, b in glat_orb_struct.items()], make_copies=False)
        self.g_lat = None
        self.hopping_lat = HoppingLattice(r, hopping_r)
        self.r_cavity = [tuple(ri) for ri in r_cavity]
        self.r_cluster = [tuple(ri) for ri in r_cluster]
        self.g_cavity = {}
        self.lambd = BlockGf(name_block_generator=[[bn, GfImFreq(
            indices=b, mesh=self.mesh)] for bn, b in gcluster_orb_struct.items()], make_copies=False)
        self.gtmp = BlockGf(name_block_generator=[[bn, GfImFreq(
            indices=b, mesh=self.mesh)] for bn, b in gcluster_orb_struct.items()], make_copies=False)
        self.g_cluster = BlockGf(name_block_generator=[[bn, GfImFreq(
            indices=b, mesh=self.mesh)] for bn, b in gcluster_orb_struct.items()], make_copies=False)
        self.lambda_imp_basis = BlockGf(name_block_generator=[[bn, GfImFreq(
            indices=b, mesh=self.mesh)] for bn, b in gcluster_orb_struct.items()], make_copies=False)

    def set(self, g_imp, mu):
        """
        sets GLocal using calculate(self, mu, selfenergy, w1, w2, n_mom), uses either filling or mu
        mu can be either of blockmatrix-type or scalar
        in the case of ccdmft the selfenergy gets changed during the calculation of gloc
        """
        if not(self.filling is None):
            assert False, 'todo, filling feature not implented yet'
            #mu = self.find_and_set_mu(self.filling, se_lat, mu, self.dmu_max)
        if self.transf is not None:
            g_imp = self.transf.backtransform_g(g_imp)
        self.g_lat_initdict['g_r'] = self.calc_g_r(g_imp)
        self.g_lat = LatticeGreensfunction(**self.g_lat_initdict)
        self.g_lat.periodize()
        self.g_cluster = self.set_g_cluster(self.g_cluster, self.g_lat)
        for ri, rj in itt.product(*[self.r_cavity]*2):
            self.g_cavity[ri, rj] = self.g_lat[ri, rj].copy()
            for ra, rb in itt.product(*[self.r_cluster]*2):
                for s, b in self.g_cavity[ri, rj]:
                    b -= self.g_lat[ri, ra][s] * self.g_lat.inverse_real_space_at(ra, rb)[
                        s] * self.g_lat[rb, rj][s]
        self.lambd.zero()
        for ila, icl in self.lat_r_to_cluster.items():
            ra, rb = ila[0], ila[1]
            bo, bi, bj = ila[2], ila[3], ila[4]
            for ri, rj in itt.product(*[self.r_cavity]*2):
                self.lambd[icl[0]][icl[1], icl[2]] += self.hopping_lat[ra, ri][bo][bi, bj] * \
                    self.g_cavity[ri, rj][bo][bi, bj] * \
                    self.hopping_lat[rj, rb][bo][bi, bj]

        if self.transf is not None:
            self.lambda_imp_basis = self.transf.transform_g(self.lambd)
            self << self.transf.transform_g(self.g_cluster)
        else:
            self.lambda_imp_basis << self.lambd
            self << self.g_cluster
        return mu

    def index_cluster(self, index_lattice):
        self.lat_r_to_cluster

    def calc_g_r(self, g_imp):
        g_r = []
        for imp_to_lat in self.imp_to_lat_r:
            self.g_lat_tmp.zero()
            for i_imp, i_lat in imp_to_lat.items():
                i0, i1, i2 = i_lat
                j0, j1, j2 = i_imp
                self.g_lat_tmp[i0][i1, i2] << g_imp[j0][j1, j2]
            g_r.append(self.g_lat_tmp.copy())
        return g_r

    def set_g_cluster(self, g_c, g_l):
        for i_lat, i_cluster in self.lat_r_to_cluster.items():
            i0, i1, i2 = i_cluster
            jr0, jr1, j0, j1, j2 = i_lat
            g_c[i0][i1, i2] << g_l[jr0, jr1][j0][j1, j2]
        return g_c


class SelfEnergy(SelfEnergyCommon):
    pass


class WeissField(WeissFieldCommon):
    def __init__(self, tcluster, *args, **kwargs):
        self.tcluster = tcluster
        WeissFieldCommon.__init__(self, *args, **kwargs)

    def calc_selfconsistency(self, gloc, selfenergy, mu):
        for s, b in self:
            b << inverse(iOmega_n + mu -
                         self.tcluster[s] - gloc.lambda_imp_basis[s])


class HoppingLattice:
    """
    r is a list of lattice translations
    hopping_r is of the same order as r
    """

    def __init__(self, r, hopping_r, accuracy=1e-12):
        self.r = np.array(r)
        self.h_r = [{s: np.array(b) for s, b in h.items()} for h in hopping_r]
        self.acc = accuracy

    def __getitem__(self, rirj):
        ri, rj = np.array(rirj[0]), np.array(rirj[1])
        d = np.abs(self.r - (ri-rj))
        i_min = np.argmin(d)
        if np.sum(d[i_min]) > self.acc:
            result = {s: 0*b for s, b in self.h_r[0].items()}
        else:
            result = self.h_r[i_min]
        return result
