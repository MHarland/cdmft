import matplotlib as mpl, sys, numpy as np
from pytriqs.archive import HDFArchive

from bethe.plot.cfg import plt, ax


is_offset = False
hfl = False
gn = 'maxent_results'
if hfl:
    gn += '_hfl'
n_graphs = len(sys.argv[1:])
colors = [mpl.cm.jet(float(i)/max(1,n_graphs-1)) for i in range(n_graphs)]
for archive_name, color, i in zip(sys.argv[1:], colors, range(len(sys.argv[1:]))):
    archive = HDFArchive(archive_name, 'r')
    if archive.is_group(gn):
        a_w = archive[gn]['a_w']
        mesh = archive[gn]['mesh']
        ax.plot(mesh.real, a_w + int(is_offset) * .2*i, label = '$\\mathrm{'+archive_name[:-3]+'}$', color = color)
    else:
        print archive_name+' has no maxent results'
ax.legend(fontsize = 4)
ax.set_xlabel("$\\omega$")
ax.set_ylabel("$A(\\omega)$")
if is_offset:
    ax.set_yticks([])

ax.set_ylim(bottom = 0)
#ax.set_xlim(mesh[0], mesh[-1])
#ax.set_xlim(-2,2)
#ax.set_xlim(-5,5)
#ax.set_xlim(-1,2)
ax.set_xlim(-.5,.5)
ax.plot([0,0], [0,ax.get_ylim()[1]], alpha=.75, color='gray', lw=1.)
fname = "maxent"
if hfl:
    fname += '_hfl'
plt.savefig(fname+'.pdf')
print fname+'.pdf ready'
plt.close()
