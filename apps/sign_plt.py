import matplotlib, sys, numpy as np
matplotlib.use("PDF")
from matplotlib import pyplot as plt

from bethe.h5interface import Storage


for fname in sys.argv[1:]:
    fig = plt.figure()
    ax = fig.add_axes([.12,.12,.75,.8])
    sto = Storage(fname)
    y = []
    x = []
    n_loops = sto.get_completed_loops()
    for l in range(n_loops):
        y.append(sto.load("average_sign", l))
        x.append(l)
    ax.plot(x, y, marker = "+")
    ax.set_xlabel("$\mathrm{DMFT-Loop}$")
    ax.set_ylabel("$<\\mathrm{sign}>$")
    ax.set_ylim(0,1.05)
    plt.savefig(fname[:-3]+"_sign.pdf")
    print fname[:-3]+"_sign.pdf ready"
    plt.close()
