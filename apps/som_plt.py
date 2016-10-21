import matplotlib as mpl, sys, numpy as np
mpl.use("PDF")
from matplotlib import pyplot as plt
from pytriqs.gf.local import GfReFreq
from pytriqs.archive import HDFArchive
from scipy.signal import savgol_filter


fig = plt.figure()
ax = fig.add_axes([.12,.1,.83,.83])
n_graphs = len(sys.argv[1:])
colors = [mpl.cm.jet(float(i)/max(1,n_graphs-1)) for i in range(n_graphs)]
for archive_name, color in zip(sys.argv[1:], colors):
    archive = HDFArchive(archive_name, 'r')
    g_w = archive['som_results']['g_w']
    mesh = np.array([w for w in g_w.mesh])
    a = -g_w.data[:,0,0].imag/np.pi
    a = savgol_filter(a, 51, 1)
    ax.plot(mesh.real, a, label = archive_name[:-3], color = color)
ax.legend(fontsize = 8)
ax.set_xlabel("$\\omega$")
ax.set_ylabel("$A(\\omega)$")
ax.set_ylim(bottom = 0)
#ax.set_xlim(-1,1)
plt.savefig("som.pdf")
print "som.pdf ready"
plt.close()
