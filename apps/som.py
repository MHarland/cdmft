import sys
from pytriqs.archive import HDFArchive
from pytriqs.gf import GfReFreq, GfLegendre, GfImTime, rebinning_tau, BlockGf, MatsubaraToLegendre, LegendreToMatsubara, GfImFreq, inverse
from pytriqs.utility import mpi
from triqs_som.som import Som
import numpy as np
from time import time
from scipy.interpolate import interp1d

from cdmft.h5interface import Storage
from cdmft.gfoperations import trace, cut_coefficients


#fnames = sys.argv[2:]
fnames = sys.argv[1:]
nambu = False
hfl = False
domain = "legendre"
npts = None
s_by = "const"
#s_by = "envelope"
nptss = [None] * len(fnames)  # [46,48,48,50,52,52,52,60,64,70,78]
run_params = {}
# dSC
"""
run_params['energy_window'] = (-24,24)
run_params['max_time'] = -1
run_params['verbosity'] = 2
run_params['t'] = 30
run_params['f'] = 30000
run_params['min_rect_width'] = 3e-4
run_params['max_rects'] = 1000
run_params['adjust_f'] = False
run_params['l'] = 40#100
run_params['adjust_l'] = False
run_params['make_histograms'] = True
run_params['hist_max'] = 10
run_params['hist_n_bins'] = 300
"""
#run_params['energy_window'] = (-int(sys.argv[1]), int(sys.argv[1]))
run_params['energy_window'] = (-16, 16)
run_params['max_time'] = -1
run_params['verbosity'] = 2
run_params['t'] = 50
run_params['f'] = 3000
run_params['adjust_f'] = False
run_params['l'] = 32  # 100
run_params['adjust_l'] = False
run_params['make_histograms'] = True
run_params['hist_max'] = 10
run_params['hist_n_bins'] = 300
for archive_name, npts in zip(fnames, nptss):
    print 'doing', archive_name
    start_time = time()
    sto = Storage(archive_name)
    if domain == "tau":
        g = sto.load("g_tau")
        if npts is not None:
            g = BlockGf(name_block_generator=[
                        (s, rebinning_tau(b, npts)) for s, b in g], make_copies=False)
        npts = len([x for x in g.mesh])
        tr_g = GfImTime(indices=range(1), beta=g.mesh.mesh.beta, n_points=npts)
        trace(g, tr_g)
        s = tr_g.copy()
    elif domain == "legendre":
        g = sto.load("g_sol_l")
        if npts is None:
            npts = len(g.mesh)
        tr_g = GfLegendre(indices=range(
            1), beta=g.mesh.mesh.beta, n_points=npts)
        if npts is not None:
            g = BlockGf(name_block_generator=[
                        (s, cut_coefficients(b, npts)) for s, b in g], make_copies=False)
        npts = len([x for x in g.mesh])
        if nambu:
            g = sto.load("g_imp_iw")
            if hfl:
                se = sto.load("se_imp_iw")
                for bn, b in se:
                    b[0, 1] << 0.
                    b[1, 0] << 0.
                g0 = sto.load("g0_iw")
                for bn, b in g0:
                    b[0, 1] << 0.
                    b[1, 0] << 0.
                g << inverse(inverse(g0)-se)
            tr_giw = GfImFreq(indices=[0], n_points=1025, beta=g.mesh.beta)
            i_ = 0
            for s, b in g:
                for i in b.indices:
                    i = int(i)
                    if i % 2:
                        tr_giw += (-1) * b[i, i].conjugate()
                    else:
                        tr_giw += b[i, i]
                    i_ += 1
            tr_giw << tr_giw / i_
            tr_g << MatsubaraToLegendre(tr_giw)
            tr_g.data[:, :, :] = tr_g.data[:, :, :].real
        else:
            trace(g, tr_g)
        # tr_g << g['XY'][1,0] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        s = tr_g.copy()
        if s_by == "envelope":
            inds = [i for i in range(0, s.data.shape[0], 2)]
            if s.data.shape[0] % 2 == 0:
                inds.append(inds[-1] + 1)
            envelope = interp1d(inds, s.data[inds, 0, 0].real)
            s.data[:, 0, 0] = np.array([envelope(i)
                                        for i in range(s.data.shape[0])])
    if s_by == "const":
        s.data[:, 0, 0] = 1.0
    g_rec = tr_g.copy()
    gw = GfReFreq(window=(run_params['energy_window'][0],
                          run_params['energy_window'][1]), n_points=5000, indices=tr_g.indices)
    som = Som(tr_g, s, kind="FermionGf")
    som.run(**run_params)
    g_rec << som
    gw << som
    if mpi.is_master_node():
        arch = HDFArchive(archive_name, 'a')
        results_groupname = 'som_results'
        if hfl:
            results_groupname += '_hfl'
        if not arch.is_group(results_groupname):
            arch.create_group(results_groupname)
        res = arch[results_groupname]
        res['g_in'] = tr_g
        res['g_rec'] = g_rec
        res['g_w'] = gw
        res['s'] = s
        if run_params['make_histograms']:
            res['histograms'] = som.histograms
        res['parameters'] = run_params
        res["calculation_time"] = time() - start_time
