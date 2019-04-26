import matplotlib as mpl, sys, numpy as np
from pytriqs.archive import HDFArchive

from cdmft.plot.cfg import plt, ax


is_offset = False
groupname = 'maxent_orbs_results'
for archive_name in sys.argv[1:]:
    archive = HDFArchive(archive_name, 'r')
    if archive.is_group(groupname):
        aw = {}
        res = archive[groupname]['a_w']
        for key, val in res.items():
            aw[key] = val
        mesh = archive[groupname]['mesh']
        n_graphs = len(aw)
        colors = [mpl.cm.jet(float(i)/max(1,n_graphs-1)) for i in range(n_graphs)]
        for (bn, a_w), color in zip(aw.items(), colors):
            if 'dn' in bn: continue
            ax.plot(mesh.real, a_w + int(is_offset) * .2*i, label = '$\\mathrm{'+bn+'}$', color = color)
    else:
        print archive_name+' has no maxent orbs results'
    ax.legend(fontsize = 6, loc = 'best')
    ax.set_xlabel("$\\omega$")
    ax.set_ylabel("$A(\\omega)$")
    if is_offset:
        ax.set_yticks([])
    ax.set_ylim(bottom = 0)
    ax.set_xlim(mesh[0], mesh[-1])
    ax.set_xlim(-.5,.5)
    #ax.set_xlim(-.4,.4)
    #ax.set_xlim(-2,2)
    outname = archive_name[:-3]+'_maxent_orbs.pdf'
    plt.savefig(outname)
    print outname+" ready"
    plt.gca().cla()
