import matplotlib
import sys
import numpy as np

from cdmft.h5interface import Storage
from cdmft.plot.cfg import plt, ax
from pytriqs.gf import inverse


index = None
w_max = 10
orb_nr = 1
for fname in sys.argv[1:]:
    sto = Storage(fname)
    n_loops = min(10, sto.get_completed_loops())
    colors = [matplotlib.cm.jet(i/float(max(1, n_loops-1)))
              for i in range(n_loops)]
    for l, c in zip(range(-n_loops, 0), colors):
        se = sto.load("se_imp_iw", l)
        g_weiss = sto.load("g0_iw", l)
        diff = se.copy()
        diff << inverse(g_weiss) - se
        if index is None:
            b, i, j = [(b, i, j) for b, i, j in se.all_indices][orb_nr]
            index_label = str(b)+str(i)+str(j)
        supermesh = np.array([iw.imag for iw in se.mesh])
        n_iw0 = int(len(supermesh)*.5)
        n_w_max = np.argwhere(supermesh <= w_max)[-1, 0]
        mesh = supermesh[n_iw0:n_w_max]
        y_a = diff[b][i, j].data[n_iw0:n_w_max, 0, 0].imag
        ax.plot(mesh, y_a, color=c, ls="--", marker="x")
    ax.set_xlabel("$i\\omega_n$")
    ax.set_ylabel("$\\Im \\Sigma^{"+b+"}_{"+str(i)+str(j)+"}(i\\omega_n)$")
    legend = ax.legend(fontsize=8, loc="lower right", title='$\\mathrm{loop}$')
    plt.setp(legend.get_title(), fontsize=8)
    ax.set_xlim(left=0)
    plt.savefig(fname[:-3]+"_loc_imp.pdf")
    print fname[:-3]+"_loc_imp.pdf ready"
    plt.cla()
