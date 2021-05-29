import matplotlib, sys, numpy as np

from cdmft.h5interface import Storage
from cdmft.plot.cfg import plt, ax


index = None
w_max = 10
orb_nr = 1
for fname in sys.argv[1:]:
    sto = Storage(fname)
    n_loops = min(10, sto.get_completed_loops())
    g_loc = sto.load("g_loc_iw")
    g_imp = sto.load("g_imp_iw")
    n_inds = len([i for i in g_loc.all_indices])
    colors = [matplotlib.cm.jet(i/float(max(1,n_inds-1))) for i in range(n_inds)]
    for (b, i, j), col in zip(g_loc.all_indices, colors):
        i, j = i[0], j[0]
        index_label = str(b)+str(i)+str(j)
        supermesh = np.array([iw.imag for iw in g_loc.mesh])
        n_iw0 = int(len(supermesh)*.5)
        n_w_max = np.argwhere(supermesh <= w_max)[-1,0]
        mesh = supermesh[n_iw0:n_w_max]
        y_a = g_loc[b][i, j].data[n_iw0:n_w_max].real
        y_b = g_imp[b][i, j].data[n_iw0:n_w_max].real
        ax.plot(mesh, y_a, color = col, ls = "--", marker = "x")
        ax.plot(mesh, y_b, color = col, label = '$'+str(b)+str(i)+str(j)+'$', marker = "+")
    ax.set_xlabel("$i\\omega_n$")
    ax.set_ylabel("$\\Im G^{"+b+"}_{"+str(i)+str(j)+"}(i\\omega_n)$")
    legend = ax.legend(fontsize = 8, loc = "lower right", title = '$\\mathrm{loop}$')
    plt.setp(legend.get_title(),fontsize=8)
    ax.set_xlim(left = 0)
    plt.savefig(fname[:-3]+"_loc_imp2.pdf")
    print fname[:-3]+"_loc_imp2.pdf ready"
    plt.cla()
