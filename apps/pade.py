import sys, numpy as np
from pytriqs.archive import HDFArchive
from pytriqs.gf.local import BlockGf, GfImTime, GfReFreq, GfImFreq
from pytriqs.utility import mpi

from bethe.h5interface import Storage
from bethe.gfoperations import trace


window = (-10, 10)
n_w = 10000
pade_n_iw = 51
nambu = False
pade_orbital = False
for archive_name in sys.argv[1:]:
    print "loading "+archive_name+"..."
    sto = Storage(archive_name)
    giw = sto.load("g_imp_iw")
    #giw = BlockGf(name_block_generator = [(s, GfImFreq(indices = [i for i in b.indices], beta = gtau.beta, n_points = n_iw))for s, b in gtau])
    #for s, b in giw:
    #    b.set_from_fourier(gtau[s])
    if nambu:
        for s, b in giw:
            for i in b.indices:
                i = int(i)
                if i%2:
                    b[i, i] << (-1) * b[i, i].conjugate()
    tr_giw = GfImFreq(indices = [0], mesh = giw.mesh)
    trace(giw, tr_giw)
    gw = BlockGf(name_block_generator = [(s, GfReFreq(indices = [i for i in b.indices], window = window, n_points = n_w))for s, b in giw])
    tr_gw = GfReFreq(indices = [0], window = window, n_points = n_w)
    eps = np.pi / giw.beta
    if pade_orbital:
        for s, b in giw:
            gw[s].set_from_pade(b, pade_n_iw, eps)
    tr_gw.set_from_pade(tr_giw, pade_n_iw, eps)
    if mpi.is_master_node():
        arch = HDFArchive(archive_name, 'a')
        if not arch.is_group('pade_results'):
            arch.create_group('pade_results')
        res = arch['pade_results']
        if pade_orbital:
            res['g_w'] = gw
        res['tr_g_w'] = tr_gw
print "pade ready"
