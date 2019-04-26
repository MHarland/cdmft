import sys

from cdmft.h5interface import Storage


for fname in sys.argv[1:]:
    print fname+':'
    sto = Storage(fname)
    nl = sto.get_completed_loops()
    for l in range(nl):
        print 'loop'+str(l)+':'
        g = sto.load('g_imp_iw', l)
        up = 0
        dn = 0
        n_blocks = 0
        for s, b in g:
            if 'up' in s:
                up += b.total_density()
            else:
                dn += b.total_density()
                n_blocks += 1
        print 'Sz = ', ((up - dn)/float(n_blocks)).real *.5
