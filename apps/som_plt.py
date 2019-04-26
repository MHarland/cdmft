import matplotlib as mpl, sys, numpy as np, os
from pytriqs.gf.local import GfReFreq
from pytriqs.archive import HDFArchive
from scipy.signal import savgol_filter

from cdmft.plot.cfg import plt, ax


#gn = 'som_scgap_results'
#gn = 'som_results'
gn = 'som_results'
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
    mesh = np.array([w for w in g_w.mesh])
    a = -g_w.data[:,0,0].imag/np.pi
    #a = savgol_filter(a, 21, 1)
    if lab is None: lab = archive_name[:-3]
    ax.plot(mesh.real, a, label = '$'+lab+'$', color = color)
ax.legend(loc = "best", frameon = False, fontsize = 6)#, title = '$U$')
ax.set_xlabel("$\\omega$")
ax.set_xlim(-5,5)
ax.set_ylabel("$A(\\omega)$")
ax.set_ylim(bottom = 0)
#ax.set_ylim(0, .9)
ax.plot([0,0], [0,ax.get_ylim()[1]], alpha=.5, color='gray')
plt.savefig("som.pdf")
print "som.pdf ready"
plt.close()
