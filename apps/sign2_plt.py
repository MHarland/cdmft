import matplotlib, sys, numpy as np

from cdmft.h5interface import Storage
from cdmft.plot.cfg import plt, ax


nc = len(sys.argv[1:])
y = []
for fname in sys.argv[1:]:
    sto = Storage(fname)
    mu = sto.load("average_sign")
    print mu
    y.append(mu)
x = range(nc)
ax.plot(x, y, marker = "+", label = '$\\mathrm{'+fname[:-3]+'}$')
ax.set_xlabel("$\mathrm{Archive}$")
ax.set_ylabel("$<\\mathrm{sign}>$")
plt.savefig("sign2.pdf")
print "sign2.pdf ready"
plt.close()
