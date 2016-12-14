import matplotlib, sys, numpy as np
matplotlib.use("PDF")
from matplotlib import pyplot as plt

from bethe.h5interface import Storage


for fname in sys.argv[1:]:
    fig = plt.figure()
    ax = fig.add_axes([.12,.12,.75,.8])
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
    ax.plot(x, y0, marker = "x", label = "density of G_loc")
    ax.plot(x, y, marker = "+", label = "density of G_imp")
    ax.set_xlabel("$\mathrm{DMFT-Loop}$")
    ax.set_ylabel("$N$")
    ax.legend()
    plt.savefig(fname[:-3]+"_density.pdf")
    print fname[:-3]+"_density.pdf ready"
    plt.close()
