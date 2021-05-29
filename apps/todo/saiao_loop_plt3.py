import matplotlib
import sys
import numpy as np
import itertools as itt

from cdmft.h5interface import Storage
from cdmft.plot.cfg import plt, ax
from cdmft.setups.bethelattice import TriangleAIAOBetheSetup as Setup


nc = len(sys.argv[1:])
colors = [matplotlib.cm.jet(i/float(max(1, nc-1))) for i in range(nc)]
sep = Setup(10, 0, 0, 0, 0, force_real=False)

sitetransf = sep.site_transf
def superindex(s, i): return s * 3 + i


for fname, c in zip(sys.argv[1:], colors):
    print 'file', fname
    sto = Storage(fname)
    yloops = []
    n_loops = sto.get_completed_loops()
    for l in range(n_loops):
        print 'loop', l
        g = sto.load('g_imp_iw', l)
        n = np.empty([6, 6])
        bname = [i for i in g.indices][0]
        for i, j in itt.product(*[range(6)]*2):
            n[i, j] = g[bname][i, j].density().real
        # print 'momentum basis:'
        # print np.round(n,3)
        n_site = np.zeros([6, 6])
        for s1, s2, i1, i2 in itt.product(range(2), range(2), range(3), range(3)):
            a1, a2 = superindex(s1, i1), superindex(s2, i2)
            n_site[a1, a2] = np.sum([sitetransf[j1, i1].conjugate() * sitetransf[j2, i2] * n[superindex(
                s1, j1), superindex(s2, j2)] for j1, j2 in itt.product(range(3), range(3))], axis=0).real
        # print 'site basis:'
        # print np.round(n_site,3)
        y = []
        for p1, p2, p3 in [(0, 1, 2), (0, 2, 1), (0, 0, 0)]:
            rots = [sep.spin_transf_mat(i*2*np.pi/3.) for i in [p1, p2, p3]]
            naiao = np.zeros([6, 6])
            for s, i in itt.product(range(2), range(3)):
                a = superindex(s, i)
                naiao[a, a] = np.sum([rots[i][s, t1].conjugate() * rots[i][s, t2] * sitetransf[k1, i] * sitetransf[k2, i].conjugate() * n[superindex(
                    t1, k1), superindex(t2, k2)] for k1, k2, t1, t2 in itt.product(range(3), range(3), range(2), range(2))], axis=0).real
            saiao = np.sum([naiao[i, i] - naiao[3+i, 3+i]
                            for i in range(3)], axis=0)/6.
            sai0 = (naiao[0, 0] - naiao[3+0, 3+0])*.5
            sai1 = (naiao[1, 1] - naiao[3+1, 3+1])*.5
            sai2 = (naiao[2, 2] - naiao[3+2, 3+2])*.5
            y.append(abs(saiao))
            print 'Saiao'+str((p1, p2, p3))+':', sai0, sai1, sai2, '->', saiao
        yloops.append(y)
    y = np.array(yloops)
    x = range(y.shape[0])
    ax.plot(x, y[:, 0], marker="+",
            label='$\\mathrm{'+fname[:-3]+'}$', color=c)
    ax.plot(x, y[:, 1], marker="+", color=c, ls='dashed')
    ax.plot(x, y[:, 2], marker="+", color=c, ls='dotted')
ax.legend(loc="best", fontsize=6, framealpha=.5)
ax.set_xlabel("$\mathrm{DMFT-Loop}$")
# ax.set_ylim(-.5,.5)
ax.set_ylabel("$<S_{aiao/z}>$")
plt.savefig("saiao3.pdf")
print "saiao3.pdf ready"
plt.close()
