import matplotlib, sys, numpy as np

from cdmft.h5interface import Storage
from cdmft.plot.cfg import plt, ax


nc = len(sys.argv[1:])
colors = [matplotlib.cm.viridis(i/float(max(1,nc-1))) for i in range(nc)]
for fname, color in zip(sys.argv[1:], colors):
    sto = Storage(fname)
    n_freq = 20
    n_loops = sto.get_completed_loops()
    x = []
    y = []
    for l in range(n_loops):
        gloc = sto.load("g_loc_iw", l)
        gimp = sto.load("g_imp_iw", l)
        gdif = gloc.copy()
        gdif << gloc - gimp
        mesh = [w for w in gloc.mesh]
        niw0 = int(len(mesh) * .5)
        dif = 0
        indices = [i for i in gdif.all_indices]
        norb = len(indices)
        norm = n_freq*norb
        for ind in indices:
            bn, i, j = ind[0], int(ind[1][0]), int(ind[2][0])
            dif += np.sum(np.abs(gdif[bn][i, j].data[niw0:niw0+n_freq]))/norm
        x.append(l)
        y.append(dif)
    ax.plot(x, np.log10(y), label = '$\\mathrm{'+fname[:-3]+'}$', color = color)
ax.set_xlabel("$\mathrm{DMFT-Loop}$")
ax.set_ylabel("$\\log_{10}\\sum_{n}^{"+str(n_freq)+"}\,\\sum_{ij}\,|G^{loc}_{ij}(i\\omega_n)-G^{imp}_{ij}(i\\omega_n)|\,/\,N$")
ax.legend(loc = "best", fontsize = 6, frameon = False)
outname = "g_loc_imp_diff.pdf"
plt.savefig(outname)
print outname+" ready"
plt.close()
