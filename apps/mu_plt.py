import matplotlib, sys, numpy as np

from cdmft.h5interface import Storage
from cdmft.plot.cfg import plt, ax


nc = len(sys.argv[1:])
colors = [matplotlib.cm.jet(i/float(max(1,nc-1))) for i in range(nc)]
for fname, c in zip(sys.argv[1:], colors):
    sto = Storage(fname)
    y = []
    x = []
    n_loops = sto.get_completed_loops()
    for l in range(n_loops):
        mu = sto.load("mu", l)
        if isinstance(mu, dict):
            for s, b in mu.items():
                mu = b[0, 0]
                break
        y.append(mu)
        x.append(l)
    ax.plot(x, y, marker = "+", label = '$\\mathrm{'+fname[:-3]+'}$', color = c)
ax.legend(loc = "best", fontsize = 6)
ax.set_xlabel("$\mathrm{DMFT-Loop}$")
ax.set_ylabel("$\\mu$")
plt.savefig("mu.pdf")
print "mu.pdf ready"
plt.close()
