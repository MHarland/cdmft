import matplotlib, sys, numpy as np

from bethe.h5interface import Storage
from bethe.plot.cfg import plt, ax


nc = len(sys.argv[1:])
colors = [matplotlib.cm.jet(i/float(max(1,nc-1))) for i in range(nc)]
for fname, c in zip(sys.argv[1:], colors):
    sto = Storage(fname)
    y = []
    x = []
    n_loops = sto.get_completed_loops()
    for l in range(n_loops):
        g = sto.load('g_imp_iw', l)
        up = 0
        dn = 0
        n_blocks = 0
        for s, b in g:
            if 'up' in s:
                up += b.total_density()
            elif 'dn' in s or 'down' in s:
                dn += b.total_density()
            else:
                assert False, 'need up/dn-blocks'
            n_blocks += 1
        y.append(((up - dn)/float(n_blocks)).real)
        x.append(l)
    ax.plot(x, y, marker = "+", label = '$\\mathrm{'+fname[:-3]+'}$', color = c)
ax.legend(loc = "upper left")
ax.set_xlabel("$\mathrm{DMFT-Loop}$")
ax.set_ylabel("$<S_{z}>$")
plt.savefig("sz.pdf")
print "sz.pdf ready"
plt.close()
