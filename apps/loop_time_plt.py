import matplotlib, sys, numpy as np
matplotlib.use("PDF")
from matplotlib import pyplot as plt

from bethe.storage import LoopStorage

n = len(sys.argv[1:])
colors = [matplotlib.cm.jet(i/float(max(1,n-1))) for i in range(n)]
fig = plt.figure()
ax = fig.add_axes([.12,.12,.75,.8])
for fname, color in zip(sys.argv[1:], colors):
    print "loading "+fname+"..."
    sto = LoopStorage(fname)
    y = []
    x = []
    n_loops = sto.get_completed_loops()
    for l in range(n_loops):
        t = sto.load("loop_time", l)
        y.append(t/float(60))
        x.append(l)
    ax.plot(x, y, marker = "+", color = color, label = fname[:-3])
ax.set_xlabel("$\mathrm{DMFT-Loop}$")
ax.set_ylabel("$t[min]$")
ax.legend(loc = "lower left", fontsize = 8)
#plt.savefig(fname[:-3]+"_looptime.pdf")
plt.savefig("looptime.pdf")
#print fname[:-3]+"_looptime.pdf ready"
print "looptime.pdf ready"
plt.close()
