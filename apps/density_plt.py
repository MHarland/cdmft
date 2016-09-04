import matplotlib, sys, numpy as np
matplotlib.use("PDF")
from matplotlib import pyplot as plt

from bethe.storage import LoopStorage


for fname in sys.argv[1:]:
    fig = plt.figure()
    ax = fig.add_axes([.12,.12,.75,.8])
    sto = LoopStorage(fname)
    y = []
    x = []
    for l in range(sto.get_completed_loops()):
        try:
            y.append(sto.load("density", l))
            x.append(l)
        except KeyError:
            print "Warning, couldnt load "+fname+" loop "+str(l)
    ax.plot(x, y)
    ax.set_xlabel("$\mathrm{DMFT-Loop}$")
    ax.set_ylabel("$N$")
    plt.savefig(fname[:-3]+"_density.pdf")
    print fname[:-3]+"_density.pdf ready"
    plt.close()
