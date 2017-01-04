import matplotlib, sys, numpy as np
matplotlib.use("PDF")
from matplotlib import pyplot as plt

from bethe.h5interface import Storage


fig = plt.figure()
ax = fig.add_axes([.12,.12,.75,.8])
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
    ax.plot(x, y, marker = "+", label = fname[:-3], color = c)
ax.legend(fontsize = 8, loc = "lower center")
ax.set_xlabel("$\mathrm{DMFT-Loop}$")
ax.set_ylabel("$\\mu$")
#plt.savefig(fname[:-3]+"_mu.pdf")
plt.savefig("mu.pdf")
print "mu.pdf ready"
plt.close()
