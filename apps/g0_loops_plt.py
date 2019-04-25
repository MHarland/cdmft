import matplotlib, sys, numpy as np

from bethe.h5interface import Storage
from bethe.plot.cfg import plt, ax


index = None
w_max = 10
orb1 = ('up', 0, 0)
orb2 = ('dn', 0, 0)
for fname in sys.argv[1:]:
    sto = Storage(fname)
    n_loops = min(10, sto.get_completed_loops())
    colors = [matplotlib.cm.jet(i/float(max(1,n_loops-1))) for i in range(n_loops)]
    for l, c in zip(range(-n_loops, 0), colors):
        g_weiss = sto.load("g_weiss_iw", l)
        supermesh = np.array([iw.imag for iw in g_weiss.mesh])
        n_iw0 = int(len(supermesh)*.5)
        n_w_max = np.argwhere(supermesh <= w_max)[-1,0]
        mesh = supermesh[n_iw0:n_w_max]
        y_a = g_weiss[orb1[0]][orb1[1], orb1[2]].data[n_iw0:n_w_max,0,0].imag
        y_b = g_weiss[orb2[0]][orb1[1], orb1[2]].data[n_iw0:n_w_max,0,0].imag
        ax.plot(mesh, y_a, color = c, ls = "--", marker = "x")
        ax.plot(mesh, y_b, color = c, label = '$'+str(l)+'$', marker = "+")
    ax.set_xlabel("$i\\omega_n$")
    ax.set_ylabel("$\\Im G(i\\omega_n)$")
    legend = ax.legend(fontsize = 8, loc = "lower right", title = '$\\mathrm{loop}$')
    plt.setp(legend.get_title(),fontsize=8)
    ax.set_xlim(left = 0)
    plt.savefig(fname[:-3]+"_weiss.pdf")
    print fname[:-3]+"_weiss.pdf ready"
    plt.cla()
