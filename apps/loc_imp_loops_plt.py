import matplotlib, sys, numpy as np

from cdmft.h5interface import Storage
from cdmft.plot.cfg import plt, ax


index = None
w_max = 5
orb_nr = 0
for fname in sys.argv[1:]:
    sto = Storage(fname)
    n_loops = min(9, sto.get_completed_loops())
    #n_loops = sto.get_completed_loops()
    colors = [matplotlib.cm.jet(i/float(max(1,n_loops-1))) for i in range(n_loops)]
    for l, c in zip(range(-n_loops, 0), colors):
        g_loc = sto.load("g_loc_iw", l)
        g_imp = sto.load("g_imp_iw", l)
        #g_sol = sto.load("g_sol_iw", l)
        if index is None:
            b, i, j = [(b, i, j) for b,i,j in g_loc.all_indices][orb_nr]
        else:
            b, i, j = index
        print b, i ,j
        index_label = str(b)+str(i)+str(j)
        supermesh = np.array([iw.imag for iw in g_loc.mesh])
        n_iw0 = int(len(supermesh)*.5)
        n_w_max = np.argwhere(supermesh <= w_max)[-1,0]
        #n_w_max = n_iw0 + 5
        mesh = supermesh[n_iw0:n_w_max]
        y_a = g_loc[b][i, j].data[n_iw0:n_w_max,0,0].imag
        y_b = g_imp[b][i, j].data[n_iw0:n_w_max,0,0].imag
        #y_c = g_sol[b][i, j].data[n_iw0:n_w_max,0,0].imag
        #ax.plot(mesh, y_c, color = c, ls = ":", marker = ".")
        ax.plot(mesh, y_a, color = c, ls = "--", marker = "x")
        ax.plot(mesh, y_b, color = c, label = '$'+str(l)+'$', marker = "+")
    ax.set_xlabel("$i\\omega_n$")
    ax.set_ylabel("$\\Im G^{"+b+"}_{"+str(i)+str(j)+"}(i\\omega_n)$")
    legend = ax.legend(fontsize = 8, loc = "lower right", title = '$\\mathrm{loop}(G_{imp})$')
    plt.setp(legend.get_title(),fontsize=8)
    #ax.set_xlim(0, 2.5)
    plt.savefig(fname[:-3]+"_loc_imp.pdf")
    print fname[:-3]+"_loc_imp.pdf ready"
    plt.cla()
