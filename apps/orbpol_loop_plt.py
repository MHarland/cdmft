import matplotlib, sys, numpy as np
from scipy.stats import sem

from cdmft.h5interface import Storage
from cdmft.plot.cfg import plt, ax


nc = len(sys.argv[1:])
colors = [matplotlib.cm.jet(i/float(max(1,nc-1))) for i in range(nc)]
for fname, c in zip(sys.argv[1:], colors):
    sto = Storage(fname)
    y = []
    x = []
    n_loops = sto.get_completed_loops()
    for l in range(n_loops):
        g = sto.load('g_imp_iw', l)
        up = 0
        dn = 0
        n_blocks = 0
        for s, b in g:
            if '-c-' in s:
                up += b.total_density()
            elif '-d-' in s:
                dn += b.total_density()
            else:
                assert False, 'need -c- -d- -blocks'
            n_blocks += 1
        y.append((up - dn).real)
        x.append(l)
    print fname, np.mean(y[-6:]), sem(y[-6:])
    ax.plot(x, y, marker = "+", label = '$\\mathrm{'+fname[:-3]+'}$', color = c)
#ax.plot([0,16], [0,0], color = 'gray')
ax.legend(loc = "best", fontsize = 6, framealpha = .5)
ax.set_xlabel("$\mathrm{DMFT-Loop}$")
#ax.set_ylim(-2,2)
#ax.set_ylim(-.1,.1)
ax.set_ylabel("$<N_c - N_d>$")
plt.savefig("orbpol.pdf")
print "orbpol.pdf ready"
plt.close()
