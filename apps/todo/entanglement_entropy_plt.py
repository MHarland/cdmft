import numpy as np, sys
from scipy.linalg import logm
from getnr.getnr import get_nr

from cdmft.evaluation.common import Evaluation
from cdmft.h5interface import Storage
from cdmft.plot.cfg import plt


entropies = []
xs = []
for arch in sys.argv[1:]:
    print 'loading '+arch+'...'
    x = get_nr(arch, 'u')[0]
    sto = Storage(arch)
    ev = Evaluation(sto)
    rho = ev.get_density_matrix()
    ent = -np.trace(rho.dot(logm(rho)))
    if not np.allclose(ent.imag, 0, atol = 1e-3):
        print 'dropping non-vanishing imaginary part!'
    ent = ent.real
    print 'S = '+str(ent)
    entropies.append(ent)
    xs.append(x)
fig = plt.figure()
ax = fig.add_axes([.1,.14,.8,.75])
order = np.argsort(xs)
xs = np.array(xs)[order]
entropies = np.array(entropies)[order]
ax.plot(xs, entropies, marker = 'o', mfc = 'none', ms = 4)
ax.set_ylabel("$S_{\\mathrm{von\,Neumann}}$")
"""
ax.set_xticks(range(len(sys.argv[1:])))
ax.set_xticklabels(["$\\mathrm{"+fname[:-3]+"}$" for fname in sys.argv[1:]])
ax.set_xticklabels(["$\\mathrm{"+str(i)+"}$" for i in [0.27, 0.4, 0.8, 1.2]])
"""
ax.set_xlabel("$U$")
ax.set_title('$\\mathrm{Normalstate:}$ $\\beta=100$, $t^\prime =0.3$, $\\delta=0.25$, $t_b =0.1$')
outname = "entanglement_entropy.pdf"
plt.savefig("entanglement_entropy.pdf")
print outname+' written'
