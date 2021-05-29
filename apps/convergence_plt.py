import matplotlib, sys, numpy as np

from cdmft.h5interface import Storage
from cdmft.plot.cfg import plt


for fname in sys.argv[1:]:
    fig = plt.figure()
    ax = fig.add_axes([.12,.13,.75,.8])
    ax2 = ax.twinx()
    indices = None
    sto = Storage(fname)
    ys = []
    x = np.array(range(sto.get_completed_loops()))
    #x = np.array(range(5))
    for l in range(sto.get_completed_loops()):
    #for l in [-5,-4,-3,-2,-1]:
        g = sto.load("g_imp_iw", l)
        if indices is None:
            #indices = [str(b)+str(i)+str(j) for b,i,j in g.all_indices]
            #print indices
            indices = [(b,i,j) for b,i,j in g.all_indices]
            labels = [b+str(i)+str(j) for b,i,j in indices]
        n_iw0 = int(len([iw for iw in g.mesh])*.5)
        ys.append([g[b][i[0],j[0]].data[n_iw0] for b,i,j in indices])
    ys = np.array(ys).T
    n_y = ys.shape[0]
    colors = [matplotlib.cm.jet(i/float(max(n_y-1,1))) for i in range(n_y)]
    for y, label, color in zip(ys, labels, colors):
        ax2.plot(x, y.real, color = color, ls = ":")
        ax.plot(x, y.imag, label = label, color = color, ls = "-")
    handles, labels = ax.get_legend_handles_labels()
    legend = plt.legend(handles, labels, fontsize = 6, loc = "upper left", labelspacing = .1)
    ax.set_xlabel("$\mathrm{DMFT-Loop}$")
    ax.set_ylabel("$\\Im G(i\\omega_0)$")
    #ax.set_ylim(-1, 1)
    ax2.set_ylabel("$\\Re G(i\\omega_0)$")
    ax2.add_artist(legend)
    #ax.set_ylim(-1,0)
    #ax2.set_ylim(-1,1)
    plt.savefig(fname[:-3]+"_convergence.pdf")
    print fname[:-3]+"_convergence.pdf ready"
    plt.close()
