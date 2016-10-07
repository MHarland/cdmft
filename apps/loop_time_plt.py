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
    n_loops = sto.get_completed_loops()
    for l in range(n_loops):
        t = sto.load("loop_time", l)
        y.append(t/float(60))
        x.append(l)
    ax.plot(x, y, marker = "+")
    ax.set_xlabel("$\mathrm{DMFT-Loop}$")
    ax.set_ylabel("$t[h]$")
    plt.savefig(fname[:-3]+"_looptime.pdf")
    print fname[:-3]+"_looptime.pdf ready"
    plt.close()
