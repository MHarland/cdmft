import matplotlib, sys, numpy as np

from bethe.h5interface import Storage
from bethe.plot.cfg import plt, ax


for fname in sys.argv[1:]:
    sto = Storage(fname)
    y = []
    y0 = []
    x = []
    n_loops = sto.get_completed_loops()
    #colors = [matplotlib.cm.jet(i/max(1,float(n_loops-1))) for i in range(n_loops)]
    for l in range(n_loops):
        y.append(sto.load("density", l))
        x.append(l)
        y0.append(sto.load("density0", l))
    ax.plot(x, y0, marker = "x", label = "$G_{loc}$")
    ax.plot(x, y, marker = "+", label = "$G_{imp}$")
    ax.set_xlabel("$\mathrm{DMFT-Loop}$")
    ax.set_ylabel("$N$")
    ax.legend(loc = "upper right")
    plt.savefig(fname[:-3]+"_density.pdf")
    print fname[:-3]+"_density.pdf ready"
    plt.cla()
