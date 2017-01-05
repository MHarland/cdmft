import matplotlib as mpl, sys, numpy as np
from pytriqs.archive import HDFArchive

from bethe.plot.cfg import plt, ax


n_graphs = len(sys.argv[1:])
colors = [mpl.cm.jet(float(i)/max(1,n_graphs-1)) for i in range(n_graphs)]
for archive_name, color in zip(sys.argv[1:], colors):
    archive = HDFArchive(archive_name, 'r')
    a_w = archive['maxent_results']['a_w']
    mesh = archive['maxent_results']['mesh']
    ax.plot(mesh.real, a_w, label = archive_name[:-3], color = color)
ax.legend()
ax.set_xlabel("$\\omega$")
ax.set_ylabel("$A(\\omega)$")
ax.set_ylim(bottom = 0)
ax.set_xlim(mesh[0], mesh[-1])
plt.savefig("maxent.pdf")
print "maxent.pdf ready"
plt.close()
