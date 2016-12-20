import sys
from pytriqs.archive import HDFArchive
from pytriqs.gf.local import BlockGf, GfImTime
from pytriqs.utility import mpi
from maxent.bryanToTRIQS import MaximumEntropy

from bethe.storage import LoopStorage


ntau = 500
nomega = 1000
bandwidth = 20
sigma = 0.003
par = {"ntau": ntau,
       "nomega": nomega,
       "bandwidth": bandwidth,
       "sigma": sigma}
for archive_name in sys.argv[1:]:
    sto = LoopStorage(archive_name)
    #g = sto.load("g_tau")
    #"""
    gw = sto.load("g_imp_iw")
    g = BlockGf(name_block_generator = [(s, GfImTime(indices = [i for i in b.indices], beta = gw.beta, n_points = len(gw.mesh)*2))for s, b in gw])
    for s, b in gw:
        g[s].set_from_inverse_fourier(b)
    #"""
    maxent = MaximumEntropy(g, ntau)
    if sigma:
        maxent.calculateTotDOS(nomega, bandwidth, sigma)
    else:
        maxent.calculateTotDOS(nomega, bandwidth)
    w = maxent.getOmegaMesh()
    a = maxent.getTotDOS()
    if mpi.is_master_node():
        arch = HDFArchive(archive_name, 'a')
        if not arch.is_group('maxent_results'):
            arch.create_group('maxent_results')
        res = arch['maxent_results']
        res['mesh'] = w
        res['a_w'] = a
        res['parameters'] = par
