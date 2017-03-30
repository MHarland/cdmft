import matplotlib, sys, numpy as np, itertools as itt

from bethe.h5interface import Storage
from bethe.plot.cfg import plt, ax
from bethe.setups.bethelattice import TriangleAIAOBetheSetup as Setup


nc = len(sys.argv[1:])
colors = [matplotlib.cm.jet(i/float(max(1,nc-1))) for i in range(nc)]
sep = Setup(10, 0, 0, 0, 0)
rots = [sep.spin_transf_mat(i*2*np.pi/3.,0) for i in range(3)]
sitetransf = sep.site_transf
superindex = lambda s, i: s * 3 + i

for fname, c in zip(sys.argv[1:], colors):
    sto = Storage(fname)
    y = []
    y2 = []
    x = []
    n_loops = sto.get_completed_loops()
    for l in range(n_loops):
        g = sto.load('g_imp_iw', l)
        n = np.empty([6, 6])
        naiao = np.empty([6, 6])
        for i, j in itt.product(*[range(6)]*2):
            n[i, j] = g['spin-site'][i, j].total_density().real
        for s1, s2, i1, i2 in itt.product(range(2), range(2), range(3), range(3)):
            a1, a2 = superindex(s1, i1), superindex(s2, i2)
            naiao[a1, a2] = np.sum([rots[i1][s1, t1] * rots[i2][t2, s2].conjugate() * sitetransf[k1, i1].conjugate() * sitetransf[i2, k2] * n[superindex(t1, k1), superindex(t2, k2)] for k1, k2, t1, t2 in itt.product(range(3), range(3), range(2), range(2))], axis = 0).real
        saiao = np.sum([naiao[i, i] - naiao[3+i, 3+i] for i in range(3)], axis = 0)
        sz =  np.sum([n[i, i] - n[3+i, 3+i] for i in range(3)], axis = 0)
        y.append(saiao)
        y2.append(sz)
        x.append(l)
    ax.plot(x, y, marker = "+", label = '$\\mathrm{'+fname[:-3]+'}$', color = c)
    ax.plot(x, y2, marker = "+", label = '$\\mathrm{'+fname[:-3]+'}$', color = c, ls = 'dashed')
ax.legend(loc = "upper left")
ax.set_xlabel("$\mathrm{DMFT-Loop}$")
ax.set_ylabel("$<S_{aiao/z}>$")
plt.savefig("saiao.pdf")
print "saiao.pdf ready"
plt.close()
