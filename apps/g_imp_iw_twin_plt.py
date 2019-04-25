import matplotlib, sys, numpy as np
from pytriqs.gf.local import inverse

from bethe.h5interface import Storage
from bethe.plot.cfg import plt, fig
from bethe.plot.tools import nice_index


atol = .0001
fig.clf()
ax = fig.add_axes([.13,.12,.76,.82])
ax2 = ax.twinx()
for fname in sys.argv[1:]:
    w_max = 1
    sto = Storage(fname)
    g = sto.load("g_imp_iw")
    #g << inverse(g)
    #print g['XY'][0,1].total_density()
    #print g['XY'][0,1].tail
    supermesh = np.array([iw.imag for iw in g.mesh])
    n_iw0 = int(len(supermesh)*.5)
    n_w_max = np.argwhere(supermesh <= w_max)[-1,0]
    mesh = supermesh[n_iw0:n_w_max]
    orbs = [i for i in g.all_indices]
    #orbs = [('XY',0,2),('XY',2,0),('XY',1,3),('XY',3,1)]
    #orbs = [('GM',0,2),('GM',2,0),('GM',1,3),('GM',3,1)]
    n_orbs = len(orbs)
    colors = [matplotlib.cm.viridis(i/float(max(2,n_orbs-1))) for i in range(n_orbs)]
    for orb, c in zip(orbs, colors):
        b, i, j = orb
        if i==j: continue
        rey = g[b][i, j].data[n_iw0:n_w_max,0,0].real
        imy = g[b][i, j].data[n_iw0:n_w_max,0,0].imag
        if g[b].N1 == 1:
            label = nice_index(b)
        else:
            label = nice_index(b)+str(i)+str(j)
        if not np.allclose(imy, 0, atol = atol):
            ax.plot(mesh, imy, label = '$'+label+'$', color = c)
        else:
            print 'skip im of',b,i,j
        if not np.allclose(rey, 0, atol = atol):
            ax2.plot(mesh, rey, color = c, ls = ':')
        else:
            print 'skip re of',b,i,j
    ax.set_xlabel("$i\\omega_n$")
    ax.set_ylabel("$\\Im G(i\\omega_n)$")
    ax2.set_ylabel("$\\Re G(i\\omega_n)$")
    ax.legend(loc = "best", frameon = False)
    ax.set_xlim(0, mesh[-1])
    outname = fname[:-3]+"_g_imp_iw.pdf"
    plt.savefig(outname)
    print outname+" ready"
    ax.clear()
    ax2.clear()
