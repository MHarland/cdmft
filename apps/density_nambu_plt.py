import matplotlib, sys, numpy as np

from cdmft.h5interface import Storage
from cdmft.plot.cfg import plt, ax
from cdmft.schemes.bethe import GLocalAFMNambu


for fname in sys.argv[1:]:
    sto = Storage(fname)
    y = []
    y0 = []
    x = []
    n_loops = sto.get_completed_loops()
    #colors = [matplotlib.cm.jet(i/max(1,float(n_loops-1))) for i in range(n_loops)]
    for l in range(n_loops):
        gloctmp = sto.load("g_loc_iw", l)
        gloc = GLocalAFMNambu(0, 0, gf_init = gloctmp)
        gloc << gloctmp
        y.append(gloc.total_density_nambu())
        x.append(l)
        gimptmp = sto.load("g_imp_iw", l)
        gimp = GLocalAFMNambu(0, 0, gf_init = gimptmp)
        gimp << gimptmp
        y0.append(gimp.total_density_nambu())
    ax.plot(x, y0, marker = "x", label = "$G_{loc}$")
    ax.plot(x, y, marker = "+", label = "$G_{imp}$")
    #ax.set_ylim(np.min(y0)-.05, np.max(y0)+.05)
    ax.set_xlabel("$\mathrm{DMFT-Loop}$")
    ax.set_ylabel("$N$")
    ax.legend(loc = "best")
    plt.savefig(fname[:-3]+"_density.pdf")
    print fname[:-3]+"_density.pdf ready"
    plt.cla()
