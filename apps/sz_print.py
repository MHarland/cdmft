import sys

from bethe.h5interface import Storage


for fname in sys.argv[1:]:
    sto = Storage(fname)
    outstr = ''
    for l in range(-5,0,1):
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
        sz = ((up - dn)/float(n_blocks)).real *.5
        outstr += str(sz)
    print
    print fname+':'
    print 'Sz = ', outstr
