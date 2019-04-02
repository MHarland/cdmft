import matplotlib as mpl, sys, numpy as np, os
from pytriqs.gf.local import GfReFreq
from pytriqs.archive import HDFArchive
from scipy.signal import savgol_filter

from bethe.plot.cfg import plt, ax


gn = 'som_results_orb'
orbitals = [('GM', 0, 0), ('XY', 0, 0), ('GM', 2, 2)]
lss = ['--', '-', ':']
n_files = len(sys.argv[1:])
colors = [mpl.cm.jet(float(i)/max(2,n_files-1)) for i in range(n_files)]
labels = [None] * n_files
for archive_name, color, lab in zip(sys.argv[1:], colors, labels):
    if not os.path.isfile(archive_name):
        print 'archive '+archive_name+' does not exist'
        continue
    archive = HDFArchive(archive_name, 'r')
    try:
        g_w = archive[gn]['g_w']
    except KeyError:
        print 'no '+gn+' in '+archive_name
        continue
    for orb, ls in zip(orbitals, lss):
        bn, i, j = orb[0], int(orb[1]), int(orb[2])
        mesh = np.array([w for w in g_w[bn].mesh])
        a = -g_w[bn][i,j].data[:,0,0].imag/np.pi
        #a = savgol_filter(a, 21, 1)
        if lab is None and ls == '-':
            lab = archive_name[:-3]
            ax.plot(mesh.real, a, label = '$'+lab+'$', color = color, ls = ls)
        else:
            ax.plot(mesh.real, a, color = color, ls = ls)
ax.legend(loc = "best", frameon = False, fontsize = 6)#, title = '$U$')
ax.set_xlabel("$\\omega$")
ax.set_xlim(-4,4)
ax.set_ylabel("$A(\\omega)$")
ax.set_ylim(bottom = 0)
ax.set_ylim(0, 2)
ax.plot([0,0], [0,ax.get_ylim()[1]], alpha=.5, color='gray')
plt.savefig("som_orb.pdf")
print "som_orb.pdf ready"
plt.close()
