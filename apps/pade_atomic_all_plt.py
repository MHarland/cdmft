import matplotlib as mpl
import sys
import numpy as np
from pytriqs.gf import GfReFreq, BlockGf
from pytriqs.archive import HDFArchive

from cdmft.plot.cfg import plt, ax


n_graphs = len(sys.argv[1:])
colors = [mpl.cm.jet(float(i)/max(1, n_graphs-1)) for i in range(n_graphs)]
for archive_name, color in zip(sys.argv[1:], colors):
    archive = HDFArchive(archive_name, 'r')
    #g_w = archive['pade_atomic_results']['g_w']
    tr_g_w = archive['pade_atomic_results']['tr_g_w']
    mesh = np.array([w for w in tr_g_w.mesh])
    a = dict()
    # for s, b in g_w:
    #    a[s] = -b.data[:,0,0].imag/np.pi
    a_tot = -tr_g_w.data[:, 0, 0].imag/np.pi
    ax.plot(mesh.real, a_tot,
            label='$\\mathrm{'+archive_name[:-3]+'}$', color=color)
ax.legend(fontsize=6)
ax.set_xlabel("$\\omega$")
ax.set_ylabel("$A(\\omega)$")
ax.set_ylim(bottom=0)
ax.set_xlim(-2, 2)
plt.savefig("pade_atomic.pdf")
plt.close()
print "pade_atomic.pdf ready"
plt.close()
