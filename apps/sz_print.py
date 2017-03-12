import sys

from bethe.h5interface import Storage


for fname in sys.argv[1:]:
    sto = Storage(fname)
    g = sto.load('g_imp_iw')
    up = 0
    dn = 0
    n_blocks = 0
    for s, b in g:
        if 'up' in s:
            up += b.total_density()
        else:
            dn += b.total_density()
        n_blocks += 1
    print
    print fname+':'
    print 'Sz = ', ((up - dn)/float(n_blocks)).real
