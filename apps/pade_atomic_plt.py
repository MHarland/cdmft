import matplotlib as mpl, sys, numpy as np
mpl.use("PDF")
from matplotlib import pyplot as plt
from pytriqs.gf.local import GfReFreq, BlockGf
from pytriqs.archive import HDFArchive


fig = plt.figure()
ax = fig.add_axes([.12,.1,.83,.83])
n_graphs = len(sys.argv[1:])
colors = [mpl.cm.jet(float(i)/max(1,n_graphs-1)) for i in range(n_graphs)]
for archive_name, color in zip(sys.argv[1:], colors):
    archive = HDFArchive(archive_name, 'r')
    g_w = archive['pade_atomic_results']['g_w']
    tr_g_w = archive['pade_atomic_results']['tr_g_w']
    mesh = np.array([w for w in g_w.mesh])
    a = dict()
    for s, b in g_w:
        a[s] = -b.data[:,0,0].imag/np.pi
    a_tot = -tr_g_w.data[:,0,0].imag/np.pi
    ax.plot(mesh.real, a_tot, label = archive_name[:-3], color = color)
ax.legend(fontsize = 8)
ax.set_xlabel("$\\omega$")
ax.set_ylabel("$A(\\omega)$")
ax.set_ylim(bottom = 0)
plt.savefig("pade_atomic.pdf")
plt.close()
print "pade_atomic.pdf ready"
plt.close()

for archive_name, color in zip(sys.argv[1:], colors):
    fig2 = plt.figure()
    ax2 = fig2.add_axes([.12,.1,.83,.83])
    archive = HDFArchive(archive_name, 'r')
    g_w = archive['pade_atomic_results']['g_w']
    tr_g_w = archive['pade_atomic_results']['tr_g_w']
    mesh = np.array([w for w in g_w.mesh])
    a = dict()
    for s, b in g_w:
        a[s] = -b.data[:,0,0].imag/np.pi
    a_tot = -tr_g_w.data[:,0,0].imag/np.pi
    n2 = len(a)
    colors2 = [mpl.cm.jet(float(i)/max(1,n2-1)) for i in range(n2)]
    for (s, a_orb), c in zip(a.items(), colors2):
        ax2.plot(mesh.real, a_orb, label = s, color = c)
    ax2.plot(mesh.real, a_tot, label = "tot", color = "gray", ls = ":")
    ax2.legend(fontsize = 8)
    ax2.set_xlabel("$\\omega$")
    ax2.set_ylabel("$A(\\omega)$")
    ax2.set_ylim(bottom = 0)
    plt.savefig(archive_name[:-3]+"_pade_atomic.pdf")
    plt.close()
    print archive_name[:-3]+"pade_atomic.pdf ready"
