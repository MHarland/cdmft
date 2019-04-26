import matplotlib, sys, numpy as np, itertools as itt

from cdmft.h5interface import Storage
from cdmft.plot.cfg import plt, ax
from cdmft.setups.bethelattice import TriangleAIAOBetheSetup as Setup


nc = len(sys.argv[1:])
colors = [matplotlib.cm.jet(i/float(max(1,nc-1))) for i in range(nc)]
sep = Setup(10, 0, 0, 0, 0, force_real = False)
rots = [sep.spin_transf_mat(i*2*np.pi/3.+np.pi) for i in [0,1,2]]
sitetransf = sep.site_transf
superindex = lambda s, i: s * 3 + i

for fname, c in zip(sys.argv[1:], colors):
    sto = Storage(fname)
    y = []
    y2 = []
    z = []
    u = []
    x = []
    n_loops = sto.get_completed_loops()
    for l in range(n_loops):
        g = sto.load('g_imp_iw', l)
        
        n = np.empty([6, 6])
        bname = [i for i in g.indices][0]
        for i, j in itt.product(*[range(6)]*2):
            n[i, j] = g[bname][i, j].total_density().real
        print 'momentum basis:'
        print np.round(n,3)
        
        n_site = np.zeros([6, 6])
        for s1, s2, i1, i2 in itt.product(range(2), range(2), range(3), range(3)):
            a1, a2 = superindex(s1, i1), superindex(s2, i2)
            n_site[a1, a2] = np.sum([sitetransf[j1, i1].conjugate() * sitetransf[j2, i2] * n[superindex(s1, j1), superindex(s2, j2)] for j1, j2 in itt.product(range(3), range(3))], axis = 0).real
        print 'site basis:'
        print np.round(n_site,3)
        
        naiao = np.zeros([6, 6])
        for s, i in itt.product(range(2), range(3)):
            a = superindex(s, i)
            naiao[a, a] = np.sum([rots[i][s, t1].conjugate() * rots[i][s, t2] * sitetransf[k1, i] * sitetransf[k2, i].conjugate() * n[superindex(t1, k1), superindex(t2, k2)] for k1, k2, t1, t2 in itt.product(range(3), range(3), range(2), range(2))], axis = 0).real
        
        saiao = np.sum([naiao[i, i] - naiao[3+i, 3+i] for i in range(3)], axis = 0)/6.
        sai0 = (naiao[0, 0] - naiao[3+0, 3+0])*.5
        sai1 = (naiao[1, 1] - naiao[3+1, 3+1])*.5
        sai2 = (naiao[2, 2] - naiao[3+2, 3+2])*.5
        sz = np.sum([n[i, i] - n[3+i, 3+i] for i in range(3)], axis = 0)/6.
        ns = [n_site[i, i] + n_site[3+i, 3+i] for i in range(3)]
        print 'N:', ns, np.sum(ns)
        print 'Saiao:', sai0, sai1, sai2, '->', saiao
        print 'Sz:', sz
        y.append(saiao)
        y2.append(sz)
        x.append(l)
    ax.plot(x, y, marker = "+", label = '$\\mathrm{'+fname[:-3]+'}$', color = c)
    ax.plot(x, y2, marker = "+", color = c, ls = 'dashed')
ax.legend(loc = "lower left", fontsize = 6)
ax.set_xlabel("$\mathrm{DMFT-Loop}$")
#ax.set_ylim(-.5,.5)
ax.set_ylabel("$<S_{aiao/z}>$")
plt.savefig("saiao2.pdf")
print "saiao2.pdf ready"
plt.close()
