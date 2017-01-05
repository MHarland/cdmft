import matplotlib, sys, numpy as np
from pytriqs.gf.local import InverseFourier, GfImTime, BlockGf

from bethe.h5interface import Storage
from bethe.plot.cfg import plt, ax


only_diag = True
for arch_name in sys.argv[1:]:
    sto = Storage(arch_name)
    giw = sto.load("g_imp_iw")
    g = BlockGf(name_block_generator = [(s, GfImTime(beta = giw.beta, n_points = 3000, indices = [i for i in b.indices])) for s, b in giw])
    for s, b in giw:
        g[s] = InverseFourier(b)
    inds = [(b, i, j) for b, i, j in g.all_indices if i == j or not only_diag]
    colors = [matplotlib.cm.jet(i/float(max(1,len(inds)-1))) for i in range(len(inds))]
    mesh = [t.real for t in g[inds[0][0]].mesh]
    for (b, i, j), color in zip(inds, colors):
        ax.plot(mesh[::10], g[b].data[::10,int(i),int(j)].real, color = color, label = "$"+b+"_{"+str(i)+str(j)+"}$")
    outname = arch_name[:-3]+"_g_tau_allorbs.pdf"
    ax.legend(fontsize = 8, loc = "lower center")
    ax.set_xlabel("$\\tau$")
    ax.set_ylabel("$G(\\tau)$")
    plt.savefig(outname)
    plt.close()
    print outname+' ready'
