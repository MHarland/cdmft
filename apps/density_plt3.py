import matplotlib, sys, numpy as np
from scipy.stats import sem

from cdmft.h5interface import Storage
from cdmft.plot.cfg import plt, ax

colors = [matplotlib.cm.jet(i/max(1.,float(len(sys.argv[1:])-1))) for i in range(len(sys.argv[1:]))]
for fname, c in zip(sys.argv[1:], colors):
    sto = Storage(fname)
    y = []
    y0 = []
    x = []
    n_loops = sto.get_completed_loops()
    #
    for l in range(n_loops):
        y.append(sto.load("density", l))
        x.append(l)
        y0.append(sto.load("density0", l))
    ax.plot(x, y0, marker = "x", ls = ':', color = c)
    ax.plot(x, y, marker = "+", label = '$'+fname[:-3]+'$', color = c)
    #ax.set_ylim(np.min(y0)-.05, np.max(y0)+.05)
    print fname, np.mean(np.array(y[-6:])), sem(np.array(y[-6:]))
#ax.set_ylim(2.9,3.2)
#ax.set_ylim(.9,1.1)
ax.set_xlabel("$\mathrm{DMFT-Loop}$")
ax.set_ylabel("$N$")
ax.legend(loc = "best", fontsize = 6)
plt.savefig("density.pdf")
print "density.pdf ready"
plt.cla()
