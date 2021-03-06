import sys
import numpy as np
from pytriqs.archive import HDFArchive
from pytriqs.gf import BlockGf, GfImTime, GfReFreq, GfImFreq
from pytriqs.utility import mpi

from cdmft.h5interface import Storage
from cdmft.gfoperations import trace


window = (-10, 10)
n_w = 10000
n_iw = 1025
pade_n_iw = 101
nambu = False
orbital = False
for archive_name in sys.argv[1:]:
    print "loading "+archive_name+"..."
    sto = Storage(archive_name)
    gtau = sto.load("atomic_gf")
    giw = BlockGf(name_block_generator=[(s, GfImFreq(
        indices=[i for i in b.indices], beta=gtau.mesh.beta, n_points=n_iw))for s, b in gtau], make_copies=False)
    for s, b in giw:
        b.set_from_fourier(gtau[s])
    if nambu:
        for s, b in giw:
            for i in b.indices:
                i = int(i)
                if i % 2:
                    b[i, i] << (-1) * b[i, i].conjugate()
    tr_giw = GfImFreq(indices=[0], mesh=giw.mesh)
    trace(giw, tr_giw)
    gw = BlockGf(name_block_generator=[(s, GfReFreq(
        indices=[i for i in b.indices], window=window, n_points=n_w))for s, b in giw], make_copies=False)
    tr_gw = GfReFreq(indices=[0], window=window, n_points=n_w)
    eps = np.pi / giw.mesh.beta
    if orbital:
        for s, b in giw:
            gw[s].set_from_pade(b, pade_n_iw, eps)
    tr_gw.set_from_pade(tr_giw, pade_n_iw, eps)
    if mpi.is_master_node():
        arch = HDFArchive(archive_name, 'a')
        if not arch.is_group('pade_atomic_results'):
            arch.create_group('pade_atomic_results')
        res = arch['pade_atomic_results']
        if orbital:
            res['g_w'] = gw
        res['tr_g_w'] = tr_gw
print "pade ready"
