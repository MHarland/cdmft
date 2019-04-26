import matplotlib, sys, numpy as np

from cdmft.h5interface import Storage
from cdmft.plot.cfg import plt, ax


n_loops_ave = 8
y = []
x = []
for fname in sys.argv[1:]:
    sto = Storage(fname)
    n_loops = sto.get_completed_loops()
    ave = 0
    if n_loops < n_loops_ave: continue
    for l in range(-1*n_loops_ave, 0, 1):
        g = sto.load('g_imp_iw', l)
        up = 0
        dn = 0
        n_blocks = 0
        for s, b in g:
            if 'up' in s:
                up += b.total_density()
            elif 'dn' in s or 'down' in s:
                dn += b.total_density()
            else:
                assert False, 'need up/dn-blocks'
            n_blocks += 1
        ave += (up - dn).real*.5 / n_loops_ave
    x.append(float(fname[-5:-3]))
    y.append(ave)
order = np.argsort(x)
x = np.array(x)[order]
y = np.abs(np.array(y)[order])
print y
ax.plot(x, y, marker = "+")
#ax.set_xticklabels([s[-5:-3] for s in sys.argv[1:]])
ax.set_ylim(0, 1.5)
ax.set_ylabel("$<S_{z}>$")
plt.savefig("sz.pdf")
print "sz.pdf ready"
plt.close()
