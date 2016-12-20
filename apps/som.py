import sys
from pytriqs.archive import HDFArchive
from pytriqs.gf.local import GfReFreq, GfLegendre, GfImTime, rebinning_tau, BlockGf
from pytriqs.utility import mpi
from triqs_som.som import Som
import numpy as np
from time import time
from scipy.interpolate import interp1d

from bethe.h5interface import Storage
from bethe.gfoperations import trace, cut_coefficients


domain = "legendre"
npts = 40
s_by = "envelope"

run_params = {}
run_params['energy_window'] = (-8, 8)
run_params['max_time'] = -1
run_params['verbosity'] = 2
run_params['t'] = 1000
run_params['f'] = 100
run_params['adjust_f'] = True
run_params['l'] = 1000
run_params['adjust_l'] = False
run_params['make_histograms'] = True
run_params['hist_max'] = 2
run_params['hist_n_bins'] = 100
for archive_name in sys.argv[1:]:
    start_time = time()
    sto = Storage(archive_name)
    if domain == "tau":
        g = sto.load("g_tau")
        if npts is not None: g = BlockGf(name_block_generator = [(s, rebinning_tau(b, npts)) for s, b in g])
        npts = len([x for x in g.mesh])
        tr_g = GfImTime(indices = range(1), beta = g.beta, n_points = npts)
        trace(g, tr_g)
        s = tr_g.copy()
    elif domain == "legendre":
        g = sto.load("g_sol_l")
        if npts is not None: g = BlockGf(name_block_generator = [(s, cut_coefficients(b, npts)) for s, b in g])
        npts = len([x for x in g.mesh])
        tr_g = GfLegendre(indices = range(1), beta = g.beta, n_points = npts)
        trace(g, tr_g)
        s = tr_g.copy()
        if s_by == "envelope":
            inds = [i for i in range(0, s.data.shape[0], 2)]
            if s.data.shape[0] % 2 == 0: inds.append(inds[-1] + 1)
            envelope = interp1d(inds, s.data[inds, 0, 0].real)
            s.data[:, 0, 0] = np.array([envelope(i) for i in range(s.data.shape[0])])
    if s_by == "const": s.data[:,0,0] = 1.0
    g_rec = tr_g.copy()
    gw = GfReFreq(window = (run_params['energy_window'][0], run_params['energy_window'][1]), n_points = 5000, indices = tr_g.indices)
    som = Som(tr_g, s, kind = "FermionGf")
    som.run(**run_params)
    g_rec << som
    gw << som
    if mpi.is_master_node():
        arch = HDFArchive(archive_name, 'a')
        results_groupname = 'som_results'
        if not arch.is_group(results_groupname):
            arch.create_group(results_groupname)
        res = arch[results_groupname]
        res['g_in'] = tr_g
        res['g_rec'] = g_rec
        res['g_w'] = gw
        res['s'] = s
        res['histograms'] = som.histograms
        res['parameters'] = run_params
        res["calculation_time"] = time() - start_time
