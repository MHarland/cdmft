import matplotlib
import sys
import numpy as np
from pytriqs.gf import InverseFourier, GfImTime, BlockGf

from cdmft.h5interface import Storage
from cdmft.plot.cfg import plt, ax


only_diag = False
for arch_name in sys.argv[1:]:
    print 'loading', arch_name+'...'
    sto = Storage(arch_name)
    for loop, alpha in zip([-2, -1], [.2, 1.]):
        giw = sto.load("g_imp_iw", loop)
        g = BlockGf(name_block_generator=[(s, GfImTime(
            beta=giw.mesh.beta, n_points=3001, indices=[i for i in b.indices])) for s, b in giw], make_copies=False)
        for s, b in giw:
            g[s] << InverseFourier(b)
        #inds = [(b, i, j) for b, i, j in g.all_indices if i == j or not only_diag]
        inds = [(b, i, j) for b, i, j in g.all_indices if b == 'XY' and (
            int(i), int(j)) in [(0, 2), (2, 0), (3, 1), (1, 3)]]
        #inds = [(b, i, j) for b, i, j in g.all_indices if b == 'XY' and i!=j]
        #inds = [(b, i, j) for b, i, j in g.all_indices if b == 'XY' and int(i)!=int(j)]
        colors = [matplotlib.cm.jet(i/float(max(1, len(inds)-1)))
                  for i in range(len(inds))]
        mesh = [t.real for t in g[inds[0][0]].mesh]
        i_betahalf = int(.5 * len(mesh))
        for (b, i, j), color in zip(inds, colors):
            #ax.plot(mesh[::10], g[b].data[::10,int(i),int(j)].real, color = color, label = "$"+b+"_{"+str(i)+str(j)+"}$")
            if alpha == 1.:
                ax.plot(mesh, g[b].data[:, int(i), int(
                    j)].real, color=color, label="$"+b+"_{"+str(i)+str(j)+"}$", alpha=alpha)
            else:
                ax.plot(mesh, g[b].data[:, int(i), int(j)].real,
                        color=color, alpha=alpha)
            ax.plot(mesh, g[b].data[:, int(j), int(i)].real,
                    color=color, alpha=alpha)
            # print b,i,j,np.round(g[b].data[i_betahalf, int(i), int(j)],5)
    outname = arch_name[:-3]+"_g_tau_allorbs.pdf"
    ax.legend(fontsize=8, loc="lower center")
    ax.set_xlabel("$\\tau$")
    ax.set_ylabel("$G(\\tau)$")
    plt.savefig(outname)
    plt.cla()
    print outname+' ready'
