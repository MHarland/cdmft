import sys
from pytriqs.archive import HDFArchive
from pytriqs.gf.local import GfReFreq, GfLegendre
from pytriqs.utility import mpi
from triqs_som.som import Som

from bethe.storage import LoopStorage
from bethe.gfoperations import trace


run_params = {}
run_params['energy_window'] = (-10, 10)
run_params['max_time'] = -1
run_params['verbosity'] = 2
run_params['t'] = 1000
run_params['f'] = 100
run_params['adjust_f'] = True
run_params['l'] = 500
run_params['adjust_l'] = False
run_params['make_histograms'] = True
for archive_name in sys.argv[1:]:
    sto = LoopStorage(archive_name)
    g = sto.load("g_sol_l")
    npts = len([x for x in g.mesh])
    tr_g = GfLegendre(indices = range(1), beta = g.beta, n_points = npts)
    g_rec = tr_g.copy()
    gw = GfReFreq(window = (run_params['energy_window'][0], run_params['energy_window'][1]), n_points = 2000, indices = tr_g.indices)
    trace(g, tr_g)
    s = tr_g.copy()
    s.data[:,0,0] = 1.0 # a constant makes sampling around FL more accurate
    som = Som(tr_g, s, kind = "FermionGf")
    som.run(**run_params)
    g_rec << som
    gw << som
    if mpi.is_master_node():
        arch = HDFArchive(archive_name, 'a')
        if not arch.is_group('som_results'):
            arch.create_group('som_results')
        res = arch['som_results']
        res['g_l'] = tr_g
        res['g_rec_l'] = g_rec
        res['g_w'] = gw
        res['histograms'] = som.histograms
        res['parameters'] = run_params
