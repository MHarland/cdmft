import matplotlib, sys, numpy as np
from pytriqs.statistics.histograms import Histogram

from bethe.h5interface import Storage
from bethe.plot.cfg import plt, ax


nc = len(sys.argv[1:])
colors = [matplotlib.cm.jet(i/float(max(1,nc-1))) for i in range(nc)]
for fname, c in zip(sys.argv[1:], colors):
    sto = Storage(fname)
    y = []
    x = []
    po = sto.load("perturbation_order")
    po_tot = sto.load("perturbation_order_total")
    print type(po)#, po
    for key, val in po.items():
        print key
        print dir(val)
        print type(val.data)
        print
    print type(po_tot)#, po_tot
    ax.plot(x, y, marker = "+", label = fname[:-3], color = c)
    ax.set_xlabel("$\\mathrm{Pert. Order}$")
    ax.set_ylabel("$P$")
    outname = fname[:-3]+'_pert_order.pdf'
    plt.savefig(outname)
    plt.close()
    print outname+" ready"
