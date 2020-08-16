import matplotlib
import sys
import numpy as np
from pytriqs.gf import InverseFourier, GfImTime, BlockGf

from cdmft.h5interface import Storage
from cdmft.plot.cfg import plt, ax


only_diag = False
for arch_name in sys.argv[1:]:
    sto = Storage(arch_name)
    g = sto.load("delta_tau")
    inds = [(b, i, j) for b, i, j in g.all_indices if i == j or not only_diag]
    colors = [matplotlib.cm.jet(i/float(max(1, len(inds)-1)))
              for i in range(len(inds))]
    mesh = [t.real for t in g[inds[0][0]].mesh]
    for (b, i, j), color in zip(inds, colors):
        ax.plot(mesh[::10], g[b].data[::10, int(i), int(j)].real,
                color=color, label="$"+b+"_{"+str(i)+str(j)+"}$")
    outname = arch_name[:-3]+"_delta_tau_allorbs.pdf"
    ax.legend(fontsize=8, loc="lower center")
    ax.set_xlabel("$\\tau$")
    ax.set_ylabel("$\\Delta(\\tau)$")
    plt.savefig(outname)
    plt.cla()
    print outname+' ready'
