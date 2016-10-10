import matplotlib, sys, numpy as np
matplotlib.use("PDF")
from matplotlib import pyplot as plt

from bethe.storage import LoopStorage


for fname in sys.argv[1:]:
    fig = plt.figure()
    ax = fig.add_axes([.12,.12,.85,.85])
    index = None
    w_max = 20
    orb_nr = 2
    sto = LoopStorage(fname)
    n_loops = sto.get_completed_loops()
    colors = [matplotlib.cm.jet(i/float(max(1,n_loops-1))) for i in range(n_loops)]
    for l, c in zip(range(n_loops), colors):
        g_loc = sto.load("g_loc_iw", l)
        g_imp = sto.load("g_imp_iw", l)
        if index is None:
            b, i, j = [(b, i, j) for b,i,j in g_loc.all_indices][orb_nr]
            index_label = str(b)+str(i)+str(j)
        supermesh = np.array([iw.imag for iw in g_loc.mesh])
        n_iw0 = int(len(supermesh)*.5)
        n_w_max = np.argwhere(supermesh <= w_max)[-1,0]
        mesh = supermesh[n_iw0:n_w_max]
        y_a = g_loc[b][i, j].data[n_iw0:n_w_max,0,0].imag
        y_b = g_imp[b][i, j].data[n_iw0:n_w_max,0,0].imag
        ax.plot(mesh, y_a, color = c, ls = "--", marker = "x")
        ax.plot(mesh, y_b, color = c, label = str(l), marker = "+")
    ax.set_xlabel("$i\\omega_n$")
    ax.set_ylabel("$\\Im G^{"+b+"}_{"+str(i)+str(j)+"}(i\\omega_n)$")
    ax.legend(fontsize = 8, loc = "lower right")
    ax.set_xlim(left = 0)
    plt.savefig(fname[:-3]+"_loc_imp.pdf")
    print fname[:-3]+"_loc_imp.pdf ready"
    plt.close()
