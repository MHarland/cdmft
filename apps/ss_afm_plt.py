import matplotlib, sys, numpy as np, itertools as itt

from cdmft.h5interface import Storage
from cdmft.plot.cfg import plt, ax
from cdmft.setups.bethelattice import TriangleAIAOBetheSetup as Setup


nc = len(sys.argv[1:])
colors = [matplotlib.cm.jet(i/float(max(1,nc-1))) for i in range(nc)]

for fname, c in zip(sys.argv[1:], colors):
    sto = Storage(fname)
    y = []
    y2 = []
    x = []
    n_loops = sto.get_completed_loops()
    for l in range(n_loops):
        g = sto.load('g_imp_iw', l)
        sz = .5 * (g['dn'].total_density() - g['up'].total_density()).real
        y.append(sz)
        x.append(l)
    ax.plot(x, y, marker = "+", label = '$\\mathrm{'+fname[:-3]+'}$', color = c)
ax.legend(loc = "lower left", fontsize = 6)
ax.set_xlabel("$\mathrm{DMFT-Loop}$")
ax.set_ylim(-1,1)
ax.set_ylabel("$<S_z>$")
plt.savefig("sz.pdf")
print "sz.pdf ready"
plt.close()
