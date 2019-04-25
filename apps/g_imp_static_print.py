import sys, numpy as np, itertools

from bethe.h5interface import Storage


for fname in sys.argv[1:]:
    print fname+':'
    sto = Storage(fname)
    g = sto.load('g_imp_iw')
    for s, b in g:
        print s
        dens = np.zeros(b.data.shape[1:])
        inds = range(len(dens))
        for i,j in itertools.product(inds, inds):
            dens[i,j] = b[i,j].density().real
        print np.round(dens,4)
