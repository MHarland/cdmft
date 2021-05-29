import matplotlib, sys, numpy as np

from cdmft.h5interface import Storage
from cdmft.plot.cfg import plt, ax


for fname in sys.argv[1:]:
    w_max = 20
    sto = Storage(fname)
    g = sto.load("g_imp_iw")
    supermesh = np.array([iw.imag for iw in g.mesh])
    n_iw0 = int(len(supermesh)*.5)
    n_w_max = np.argwhere(supermesh <= w_max)[-1,0]
    mesh = supermesh[n_iw0:n_w_max]
    orbs = [i for i in g.all_indices]
    n_orbs = len(orbs)
    colors = [matplotlib.cm.jet(i/float(max(1,n_orbs-1))) for i in range(n_orbs)]
    for orb, c in zip(orbs, colors):
        b, i, j = orb
        y = g[b][i[0], j[0]].data[n_iw0:n_w_max].imag
        ax.plot(mesh, y, label = '$'+b+str(i)+str(j)+'$', color = c)
    ax.set_xlabel("$i\\omega_n$")
    ax.set_ylabel("$\\Im G(i\\omega_n)$")
    ax.legend(fontsize = 8, loc = "lower right")
    ax.set_xlim(left = 0)
    outname = fname[:-3]+"_g_imp_iw.pdf"
    plt.savefig(outname)
    print outname+" ready"
    plt.cla()
