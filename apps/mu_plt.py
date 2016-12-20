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
        mu = sto.load("mu", l)
        if isinstance(mu, dict):
            for s, b in mu.items():
                mu = b[0, 0]
                break
        y.append(mu)
        x.append(l)
    ax.plot(x, y, marker = "+")
    ax.set_xlabel("$\mathrm{DMFT-Loop}$")
    ax.set_ylabel("$\\mu$")
    plt.savefig(fname[:-3]+"_mu.pdf")
    print fname[:-3]+"_mu.pdf ready"
    plt.close()
