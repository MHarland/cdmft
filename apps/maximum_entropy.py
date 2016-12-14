import sys
from pytriqs.archive import HDFArchive
from pytriqs.gf.local import BlockGf
from pytriqs.utility import mpi
from maxent.bryanToTRIQS import MaximumEntropy

from bethe.h5interface import Storage


ntau = 500
nomega = 1000
bandwidth = 20
sigma = 0.003
par = {"ntau": ntau,
       "nomega": nomega,
       "bandwidth": bandwidth,
       "sigma": sigma}
for archive_name in sys.argv[1:]:
    sto = Storage(archive_name)
    g = sto.load("g_tau")
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
