import sys
from pytriqs.archive import HDFArchive
from pytriqs.gf.local import GfReFreq, GfLegendre, GfImTime, rebinning_tau, BlockGf, MatsubaraToLegendre, LegendreToMatsubara, GfImFreq, inverse
from pytriqs.utility import mpi
from triqs_som.som import Som
import numpy as np
from time import time
from scipy.interpolate import interp1d

from bethe.h5interface import Storage
from bethe.gfoperations import trace, cut_coefficients


fnames = sys.argv[1:]
nambu = False
hfl = False
domain = "legendre"
npts = None
s_by = "const"
orbitals = [('GM', 0, 0), ('XY', 0, 0), ('GM', 2, 2)]
gimp = True
nptss = [None] * len(fnames)
run_params = {}
#dSC
run_params['energy_window'] = (-24,24)
run_params['max_time'] = -1
run_params['verbosity'] = 2
run_params['t'] = 50
run_params['f'] = 30000
run_params['min_rect_width'] = 3e-4
run_params['max_rects'] = 1000
run_params['adjust_f'] = False
run_params['l'] = 192
run_params['adjust_l'] = False
run_params['make_histograms'] = True
run_params['hist_max'] = 10
run_params['hist_n_bins'] = 300
"""
#run_params['energy_window'] = (-int(sys.argv[1]), int(sys.argv[1]))
run_params['energy_window'] = (-16,16)
run_params['max_time'] = -1
run_params['verbosity'] = 2
run_params['t'] = 50
run_params['f'] = 3000
run_params['adjust_f'] = False
run_params['l'] = 32#100
run_params['adjust_l'] = False
run_params['make_histograms'] = True
run_params['hist_max'] = 10
run_params['hist_n_bins'] = 300
"""
for archive_name, npts in zip(fnames, nptss):
    if mpi.is_master_node():
        print archive_name
    start_time = time()
    sto = Storage(archive_name)
    if gimp:
        if domain == "tau":
            g = sto.load("g_tau")
            if npts is not None: g = BlockGf(name_block_generator = [(s, rebinning_tau(b, npts)) for s, b in g])
            npts = len([x for x in g.mesh])
        elif domain == "legendre":
            g = sto.load("g_sol_l")
            if npts is None: npts = len(g.mesh)
            if npts is not None: g = BlockGf(name_block_generator = [(s, cut_coefficients(b, npts)) for s, b in g])
            npts = len([x for x in g.mesh])
            if nambu:
                g = sto.load("g_imp_iw")
                if hfl:
                    se = sto.load("se_imp_iw")
                    for bn, b in se:
                        b[0,1] << 0.
                        b[1,0] << 0.
                    g0 = sto.load("g0_iw")
                    for bn, b in g0:
                        b[0,1] << 0.
                        b[1,0] << 0.
                    g << inverse(inverse(g0)-se)

                i_ = 0
                for s, b in g:
                    for i in b.indices:
                        i = int(i)
                        if i%2:
                            b[i,i] << (-1) * b[i,i].conjugate()
                        i_ += 1
    else: # gloc
        gin = sto.load("g_loc_iw")
        if npts is None:
            npts = 101
        g = BlockGf(name_block_generator = [(bn, GfImFreq(indices = b.indices, n_points = npts, beta = gin.beta)) for bn, b in gin])
        for bn, b in g:
            new_mesh = np.array([w.imag for w in b.mesh])
            old_mesh = np.array([w.imag for w in gin[bn].mesh])
            for nm_, nm in enumerate(new_mesh):
                om_ = np.argmin(np.abs(nm - old_mesh))
                b.data[nm_, :, :] = gin[bn].data[om_, :, :]
    s = g.copy()
    if s_by == "const":
        for bn, b in s:
            b.data[:,:,:] = np.ones(b.data.shape)
    g_rec = g.copy()
    gw = BlockGf(name_block_generator = [(bn, GfReFreq(window = (run_params['energy_window'][0], run_params['energy_window'][1]), n_points = 5000, indices = b.indices)) for bn, b in g])
    for orb in orbitals:
        if mpi.is_master_node():
            print orb
        bn, i, j = orb[0], int(orb[1]), int(orb[2])
        som = Som(g[bn][i,j], s[bn][i,j], kind = "FermionGf")
        som.run(**run_params)
        g_rec[bn][i,j] << som
        gw[bn][i,j] << som
    if mpi.is_master_node():
        arch = HDFArchive(archive_name, 'a')
        results_groupname = 'som_results_orb'
        if hfl:
            results_groupname += '_hfl'
        if not gimp:
            results_groupname += '_gloc'
        if not arch.is_group(results_groupname):
            arch.create_group(results_groupname)
        res = arch[results_groupname]
        res['g_in'] = g
        res['g_rec'] = g_rec
        res['g_w'] = gw
        res['s'] = s
        if run_params['make_histograms']:
            res['histograms'] = som.histograms
        res['parameters'] = run_params
        res["calculation_time"] = time() - start_time
