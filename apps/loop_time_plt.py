import matplotlib, sys, numpy as np

from bethe.h5interface import Storage
from bethe.plot.cfg import plt, ax


n = len(sys.argv[1:])
colors = [matplotlib.cm.jet(i/float(max(1,n-1))) for i in range(n)]
for fname, color in zip(sys.argv[1:], colors):
    print "loading "+fname+"..."
    sto = Storage(fname)
    y = []
    x = []
    n_loops = sto.get_completed_loops()
    for l in range(n_loops):
        t = sto.load("loop_time", l)
        y.append(t/float(60))
        x.append(l)
    ax.plot(x, y, marker = "+", color = color, label = '$\\mathrm{'+fname[:-3]+'}$')
ax.set_xlabel("$\mathrm{DMFT-Loop}$")
ax.set_ylabel("$t[min]$")
ax.legend(loc = "upper right", fontsize = 6)
plt.savefig("looptime.pdf")
print "looptime.pdf ready"
plt.close()
