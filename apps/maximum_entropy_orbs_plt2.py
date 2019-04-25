import matplotlib as mpl, sys, numpy as np
from pytriqs.archive import HDFArchive
from mpl_to_latex.matplotlib_to_latex import PRL


offset = -1.5
groupname = 'maxent_orbs_results'
fnames = sys.argv[1:]
nf = len(fnames)
colors = [mpl.cm.jet(float(i)/max(1,nf-1)) for i in range(nf)]
orbtols = {'GM00': '--', 'GM22': ':', 'XY00': '-'}
mplcfg = PRL()
ax = mplcfg.ax

for f_, archive_name, color in zip(range(nf), fnames, colors):
    archive = HDFArchive(archive_name, 'r')
    if archive.is_group(groupname):
        aw = {}
        res = archive[groupname]['a_w']
        for key, val in res.items():
            aw[key] = val
        mesh = archive[groupname]['mesh']
        pltkwargs ={'color': color}
        plots = {}
        for bn, a_w in aw.items():
            pltkwargs['ls'] = orbtols[bn]
            if orbtols[bn] == '-':
                i = archive_name.find('mu')
                lab = archive_name[i+2:i+6]
                pltkwargs['label'] = lab
                ax.text(-3.75, offset*f_+.15, '$\mu ='+lab+'$', color = color)
            plots[orbtols[bn]] = ax.plot(mesh.real, a_w + offset*f_, **pltkwargs)
    else:
        print archive_name+' has no maxent orbs results'

ax.text(-1,1.2, '$\Gamma$', ha = 'center')
ax.text(0,2.5, '$X/Y$', ha = 'center')
ax.text(2.5,.5, '$M$', ha = 'center')
ax.set_xlabel("$\\omega$")
ax.set_ylabel("$A(\\omega)$")
if offset:
    ax.set_yticks([])
    ax.set_ylim(offset*(nf-1), 3.25)
else:
    ax.set_ylim(bottom = 0)
ax.set_xlim(mesh[0], mesh[-1])
ax.set_xlim(-4,4)
#ax.set_xlim(-.4,.4)
#ax.set_xlim(-2,2)
outname = 'maxent_orbs.pdf'
mplcfg.save(outname)

