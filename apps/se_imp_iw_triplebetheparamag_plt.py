import matplotlib, sys, numpy as np

from bethe.h5interface import Storage
from bethe.plot.cfg import plt, ax
from bethe.setups.bethelattice import TriangleBetheSetup as Setup


part = "Im"
for fname in sys.argv[1:]:
    w_max = 10
    sto = Storage(fname)
    g_mom = sto.load("se_imp_iw")
    setup = Setup(g_mom.beta, 0, 0, 0, 0)
    g = setup.transf.backtransform_g(g_mom)
    supermesh = np.array([iw.imag for iw in g.mesh])
    n_iw0 = int(len(supermesh)*.5)
    n_w_max = np.argwhere(supermesh <= w_max)[-1,0]
    mesh = supermesh[n_iw0:n_w_max]
    orbs = [i for i in g.all_indices]
    orbs = [("up",0,0), ("up",0,1)]
    n_orbs = len(orbs)
    colors = [matplotlib.cm.jet(i/float(max(1,n_orbs-1))) for i in range(n_orbs)]
    for orb, c in zip(orbs, colors):
        b, i, j = orb
        if part == "Im":
            y = g[b][i, j].data[n_iw0:n_w_max,0,0].imag
        if part == "Re":
            y = g[b][i, j].data[n_iw0:n_w_max,0,0].real
        if np.allclose(y, 0):
            print 'skipping '+b+str(i)+str(j)+' because it\'s zero'
            continue
        ax.plot(mesh, y, label = '$'+b+str(i)+str(j)+'$', color = c)
    ax.set_xlabel("$i\\omega_n$")
    ax.set_ylabel("$\\"+str(part)+"\\Sigma(i\\omega_n)$")
    leg = ax.legend(loc = "upper right")
    leg.get_frame().set_alpha(.5)
    ax.set_xlim(left = 0)
    outname = fname[:-3]+"_se_imp_site_iw.pdf"
    plt.savefig(outname)
    print outname+" ready"
    plt.cla()
