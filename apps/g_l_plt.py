import matplotlib, sys, numpy as np

from bethe.h5interface import Storage
from bethe.plot.cfg import plt, ax


for fname in sys.argv[1:]:
    sto = Storage(fname)
    loop_nr = sto.get_last_loop_nr()
    g = sto.load("g_sol_l", loop_nr)
    x = [l.real for l in g.mesh]
    orbitals = [(b, i, j) for b,i,j in g.all_indices]
    n_orb = len(orbitals)
    colors = [matplotlib.cm.jet(i/max(1,float(n_orb-1))) for i in range(n_orb)]
    for orb, c in zip(orbitals, colors):
        y = np.log(abs(g[orb[0]][orb[1],orb[2]].data[:,0,0]))
        label = str(orb[0])+str(orb[1])+str(orb[2])
        if max(y) > -30.:
            ax.plot(x, y, color = c, label = '$'+label+'$')
        else:
            print 'omitting '+label+'...'
    ax.set_xlabel("$l$")
    ax.set_ylabel("$\\mathrm{ln}\,\\mathrm{abs}\,G(l)$")
    ax.legend(fontsize = 8, loc = "upper right")
    plt.savefig(fname[:-3]+"_g_l.pdf")
    print fname[:-3]+"_g_l.pdf ready"
    plt.cla()
