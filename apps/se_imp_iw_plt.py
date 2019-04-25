import matplotlib, sys, numpy as np

from bethe.h5interface import Storage
from bethe.plot.cfg import plt, fig


def nice_index(index):
    return index.replace('dn', '\\downarrow').replace('up', '\\uparrow')

atol = .01
fig.clf()
ax = fig.add_axes([.13,.12,.78,.82])
ax2 = ax.twinx()
for fname in sys.argv[1:]:
    print 'loading',fname+'...'
    w_max = 100
    sto = Storage(fname)
    g = sto.load("se_imp_iw")
    supermesh = np.array([iw.imag for iw in g.mesh])
    n_iw0 = int(len(supermesh)*.5)
    n_w_max = np.argwhere(supermesh <= w_max)[-1,0]
    mesh = supermesh[n_iw0:n_w_max]
    orbs = [i for i in g.all_indices]
    n_orbs = len(orbs)
    colors = [matplotlib.cm.viridis(i/float(max(2,n_orbs-1))) for i in range(n_orbs)]
    for orb, c in zip(orbs, colors):
        b, i, j = orb
        rey = g[b][i, j].data[n_iw0:n_w_max,0,0].real
        imy = g[b][i, j].data[n_iw0:n_w_max,0,0].imag
        if g[b].N1 == 1:
            label = nice_index(b)
        else:
            label = nice_index(b)+str(i)+str(j)
        if not np.allclose(imy, 0, atol = atol):
            ax.plot(mesh, imy, label = '$'+label+'$', color = c)
        if not np.allclose(rey, 0, atol = atol):
            ax2.plot(mesh, rey, color = c, ls = ':')
    ax.set_xlabel("$i\\omega_n$")
    ax.set_ylabel("$\\Im\\Sigma(i\\omega_n)$")
    ax2.set_ylabel("$\\Re\\Sigma(i\\omega_n)$")
    ax.legend(loc = "best", frameon = False)
    ax.set_xlim(left = 0)
    outname = fname[:-3]+"_se_imp_iw.pdf"
    plt.savefig(outname)
    print outname+" ready"
    ax.clear()
    ax2.clear()
