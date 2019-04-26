import matplotlib as mpl, sys, numpy as np
from pytriqs.gf.local import GfReFreq, BlockGf
from pytriqs.archive import HDFArchive

from cdmft.plot.cfg import plt, ax


n_graphs = len(sys.argv[1:])
colors = [mpl.cm.jet(float(i)/max(1,n_graphs-1)) for i in range(n_graphs)]
for archive_name, color in zip(sys.argv[1:], colors):
    archive = HDFArchive(archive_name, 'r')
    g_w = archive['pade_results']['g_w']
    tr_g_w = archive['pade_results']['tr_g_w']
    mesh = np.array([w for w in g_w.mesh])
    a = dict()
    for s, b in g_w:
        for i in range(b.N1):
            a[s+str(i)+str(i)] = -b.data[:,i,i].imag/np.pi
    a_tot = -tr_g_w.data[:,0,0].imag/np.pi
    n2 = len(a)
    colors2 = [mpl.cm.jet(float(i)/max(1,n2-1)) for i in range(n2)]
    for (s, a_orb), c in zip(a.items(), colors2):
        ax.plot(mesh.real, a_orb, label = '$'+s+'$', color = c)
    ax.plot(mesh.real, a_tot, label = "$\\mathrm{tot}$", color = "black")
    ax.legend(loc = 'upper left')
    ax.set_xlabel("$\\omega$")
    ax.set_ylabel("$A(\\omega)$")
    ax.set_ylim(bottom = 0)
    ax.set_xlim(-.4,.4)
    plt.savefig(archive_name[:-3]+"_pade.pdf")
    plt.cla()
    print archive_name[:-3]+"_pade.pdf ready"
