import numpy as np, itertools as itt
from scipy.stats import sem, linregress
from pytriqs.utility import mpi
from cdmft.evaluation.common import Evaluation


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

class Criterion:
    def __init__(self, storage, loops = range(-8,0,1), n_omega = 30, lim_absgdiff = 1e-3, lim_slopes = 1e-3, lim_staticslopes = 1e-3, lim_staticsem = 3e-3, verbose = True):
        self.loops, self.n_omega, self.lim_absgdiff, self.lim_slopes, self.lim_staticslopes, self.lim_staticsem = loops, n_omega, lim_absgdiff, lim_slopes, lim_staticslopes, lim_staticsem
        self.storage = storage
        self.evaluation = Evaluation(storage)
        self.verbose = verbose

    def confirms_convergence(self):
        loops, n_omega, lim_absgdiff, lim_slopes, lim_staticslopes, lim_staticsem = self.loops, self.n_omega, self.lim_absgdiff, self.lim_slopes, self.lim_staticslopes, self.lim_staticsem
        converged = False
        gimp0 = self.storage.load("g_imp_iw")
        orbs = [threetuple for threetuple in gimp0.all_indices]
        mesh = np.array([w.imag for w in gimp0[orbs[1][0]].mesh])
        del gimp0
        meshinds = np.where(mesh >= 0 )[0][:n_omega]
        if self.storage.get_completed_loops() < len(loops):
            absgdiff = np.empty((len(orbs), len(meshinds)))
            gloc_loop = self.storage.load("g_loc_iw")
            gimp_loop = self.storage.load("g_imp_iw")
            gdiftmp = gloc_loop.copy()
            gdiftmp << gloc_loop - gimp_loop
            for (iorb, orb), (iomega, omega) in itt.product(enumerate(orbs), enumerate(meshinds)):
                b, i, j = orb[0], int(orb[1]), int(orb[2])
                absgdiff[iorb, iomega] = np.absolute(gdiftmp[b].data[omega, i, j]) / np.max((np.absolute(gimp_loop[b].data[omega, i, j]), np.absolute(gloc_loop[b].data[omega, i, j])))
            if (absgdiff[:,:] < lim_absgdiff).all():
                converged = True
            if self.verbose and mpi.is_master_node():
                print 'Convergence-Criterion:'
                print 'absgdiff <',lim_absgdiff,':', (absgdiff[:,:] < lim_absgdiff).all(), absgdiff[:,:].max()
        else:
            gdiff = np.empty((len(orbs), len(meshinds), len(loops)), dtype = complex)
            gimp = np.empty((len(orbs), len(meshinds), len(loops)))
            gimpstatic = np.empty((len(orbs), len(loops)))
            absgdiff = np.empty((len(orbs), len(meshinds), len(loops)))
            for iloop, loop in enumerate(loops):
                gloc_loop = self.storage.load("g_loc_iw", loop)
                gimp_loop = self.storage.load("g_imp_iw", loop)
                gdiftmp = gloc_loop.copy()
                gdiftmp << gloc_loop - gimp_loop
                for (iorb, orb), (iomega, omega) in itt.product(enumerate(orbs), enumerate(meshinds)):
                    b, i, j = orb[0], int(orb[1]), int(orb[2])
                    gdiff[iorb, iomega, iloop] = gdiftmp[b].data[omega, i, j] / np.max((np.absolute(gimp_loop[b].data[omega, i, j]), np.absolute(gloc_loop[b].data[omega, i, j])))
                    gimp[iorb, iomega, iloop] = np.absolute(gimp_loop[b].data[omega, i, j])
                    if iomega == 0:
                        gimpstatic[iorb, iloop] = np.absolute(gimp_loop[b][i, j].total_density())
            absgdiff = np.absolute(gdiff)
            gdiffslopes = np.array([[linregress(loops, absgdiff[iorb, iomega, :])[0] for iomega in range(len(meshinds))] for iorb in range(len(orbs))])
            gimpslopes = np.array([[linregress(loops, gimp[iorb, iomega, :])[0] for iomega in range(len(meshinds))] for iorb in range(len(orbs))])
            gimpstaticslopes = np.array([linregress(loops, gimpstatic[iorb, :])[0] for iorb in range(len(orbs))])
            gimpstaticsem = np.array([sem(gimpstatic[iorb, :]) for iorb in range(len(orbs))])
            if (absgdiff[:,:,-1] < lim_absgdiff).all() or ((np.abs(gdiffslopes) < lim_slopes).all() and (np.abs(gimpslopes) < lim_slopes).all() and (np.abs(gimpstaticslopes) < lim_staticslopes).all() and (gimpstaticsem < lim_staticsem).all()):
                converged = True
            if self.verbose and mpi.is_master_node():
                print 'Convergence-Criterion:'
                print 'absgdiff <',lim_absgdiff,':', (absgdiff[:,:,-1] < lim_absgdiff).all(), absgdiff[:,:,-1].max()
                print 'gdiffslopes <',lim_slopes,':', (np.abs(gdiffslopes) < lim_slopes).all(), np.abs(gdiffslopes).max()
                print 'gimpslopes <',lim_slopes,':', (np.abs(gimpslopes) < lim_slopes).all(), np.abs(gimpslopes).max()
                print 'gimpstaticslopes <',lim_staticslopes,':', (np.abs(gimpstaticslopes) < lim_staticslopes).all(), np.abs(gimpstaticslopes).max()
                print 'gimpstaticsem <',lim_staticsem,':', (gimpstaticsem < lim_staticsem).all(), gimpstaticsem.max()
        return converged
