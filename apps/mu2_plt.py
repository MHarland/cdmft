import matplotlib, sys, numpy as np

from bethe.h5interface import Storage
from bethe.plot.cfg import plt, ax


nc = len(sys.argv[1:])
y = []
for fname in sys.argv[1:]:
    sto = Storage(fname)
    mu = sto.load("mu")
    if isinstance(mu, dict):
        for s, b in mu.items():
            mu = b[0, 0]
            break
    y.append(mu)
x = range(nc)
ax.plot(x, y, marker = "+", label = '$\\mathrm{'+fname[:-3]+'}$')
ax.set_xlabel("$\mathrm{Archive}$")
ax.set_ylabel("$\\mu$")
plt.savefig("mu2.pdf")
print "mu2.pdf ready"
plt.close()
