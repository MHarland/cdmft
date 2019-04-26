import matplotlib, sys, numpy as np, itertools as itt
from pytriqs.gf.local import Block2Gf, GfImFreq

from cdmft.h5interface import Storage
from cdmft.plot.cfg import plt, ax


for fname in sys.argv[1:]:
    sto = Storage(fname)
    g2 = sto.load("G2_iw_inu_inup_ph", -1, bcast = False)
    inds = [i for i in g2.all_indices]
    n_inds = len(inds)
    beta = g2[inds[0][0], inds[0][1], inds[0][2], inds[0][3]].mesh.components[0].beta
    n_iw = int(len(g2[inds[0][0], inds[0][1], inds[0][2], inds[0][3]].mesh.components[0])*.5) +1
    n1 = g2._Block2Gf__indices1
    n2 = g2._Block2Gf__indices2
    colors = [plt.cm.jet(i/float(max(1, n_inds-1))) for i in range(n_inds)]
    x = [w.imag for w in g2[inds[0][0], inds[1][0]].mesh.components[0]]
    giw = Block2Gf(name_list1 = n1, name_list2 = n2, block_list = [[GfImFreq(beta = beta, indices = [0], statistic = 'Boson', n_points = n_iw) for m in n2] for n in n1])
    for s, b in g2:
        for g2_iw_inu in b.data.transpose(1,2,0,3,4,5,6):
            norm1 = beta
            for g2_iw in g2_iw_inu:
                norm2 = beta
                norm3 = len(g2_iw[0,:,:,:,:])
                norm = norm1 * norm2 * norm3
                for i in range(norm3):
                    giw[s].data[:,0,0] += g2_iw[:,i,i,i,i]/norm
    for (s, b), color in zip(giw, colors):
        y = b.data[:,0,0].real
        ax.plot(x, y, label = '$'+s[0]+s[1]+'$', color = color, marker ='+')
    ax.set_xlabel("$i\\omega_n$")
    ax.set_ylabel("$\\Re G^{(2)}(i\\omega_n)$")
    ax.legend(fontsize = 8, loc = "upper right")
    ax.set_xlim(left = 0)
    #ax.set_xlim(left = 0)
    outname = fname[:-3]+"_g2_imp_iw.pdf"
    plt.savefig(outname)
    print outname+" ready"
    plt.cla()

