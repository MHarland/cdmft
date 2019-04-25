import matplotlib, sys, numpy as np

from bethe.h5interface import Storage
from bethe.plot.cfg import plt, ax

y = []
x = []
xlabs = []
for i, fname in enumerate(sys.argv[1:]):
    print 'loading '+fname+'...'
    sto = Storage(fname)
    y.append(sto.load("density"))
    x.append(i)
    xlabs.append(fname[fname.find('tb')+2:-3])
ax.scatter(x, y, marker = "+", label = "$G_{imp}$")
ax.set_xticks(x)
ax.set_xticklabels(xlabs)
#ax.set_ylim(2.9,3.1)
ax.set_xlabel("$\mathrm{DMFT-Loop}$")
ax.set_ylabel("$N$")
plt.savefig("density.pdf")
print "density.pdf ready"
plt.cla()
