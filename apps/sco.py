import matplotlib
import sys
import numpy as np
import itertools as itt

from cdmft.h5interface import Storage
from cdmft.plot.cfg import plt, ax


for fname in sys.argv[1:]:
    print 'loading '+fname+'...'
    sto = Storage(fname)
    n_loops = sto.get_completed_loops()
    loops = range(n_loops)
    orbs = [(k, i, i) for k, i in itt.product(['G', 'M'], [0, 1])] + \
        [(k, i, j) for k, i, j in itt.product(['X', 'Y'], [0, 1], [0, 1])]
    orders = ['ntot', 'sco', 'pafm']
    nc = len(orbs)
    colors = [matplotlib.cm.jet(i/float(max(1, nc-1))) for i in range(nc)]
    graphs = {orb: [] for orb in orbs + orders}
    sco_l = []
    scoerr_l = []
    for l in loops:
        g = sto.load('g_imp_iw', l)
        ntot = 0
        for orb in orbs:
            n = 0
            b, i, j = orb
            if (i, j) == (1, 1):
                n += (-1 * g[b][i, j].conjugate()).total_density()
            if (i, j) == (0, 0):
                n += g[b][i, j].density()
            if n.imag > 0.001:
                print 'warning: non-vanishing imaginary part!'
            ntot += n.real
            graphs[orb].append(n.real)
        scoy01 = g['Y'][0, 1].density()
        scoy10 = g['Y'][1, 0].density()
        scox10 = g['X'][1, 0].density()
        scox01 = g['X'][0, 1].density()
        scos = [np.abs(sc) for sc in [scoy01, scoy10, scox10, scox01]]
        sco = np.mean(scos)
        sco_l.append(sco)
        scoerr = np.max(np.abs(scos - sco))
        scoerr_l.append(scoerr)
        afm = np.sum([g[k][0, 0].density() - (-1) * g[k][1, 1].conjugate().density()
                      for k in ['G', 'X', 'Y', 'M']]) * .5
        for key, val in {'ntot': ntot, 'sco': sco, 'pafm': afm}.items():
            graphs[key].append(val.real)
    print 'x('+str(l)+') = '+str(1-.25*ntot.real)
    print 'sco('+str(l)+') = '+str(sco.real)
    # print 'afm('+str(l)+') = '+str(afm.real)

    for order in ['sco']:  # orders:
        kwargs = {'label': "$\\mathrm{"+order+"}$"}
        ax.plot(loops, graphs[order], **kwargs)
    ax.errorbar(loops, sco_l, yerr=scoerr_l)
    ax.legend(loc="best", fontsize=6)
    ax.set_xlabel("$\mathrm{DMFT-Loop}$")
    ax.set_ylabel("$<\\mathrm{orderparameter}$>")
    ax.set_ylim(-.01, .06)
    outname = fname[:-3]+"_sco1.pdf"
    plt.savefig(outname)
    print outname, "ready"
    ax.clear()

    for orb, c in zip(orbs, colors):
        b, i, j = orb
        kwargs = {'label': '$\\mathrm{'+b+'}'+str(i)+str(j)+'$', 'color': c}
        if i != j:
            kwargs['ls'] = '--'
        ax.plot(loops, graphs[orb], **kwargs)
    ax.legend(loc="best", fontsize=6)
    ax.set_xlabel("$\mathrm{DMFT-Loop}$")
    ax.set_ylabel("$n$")
    outname = fname[:-3]+"_sco2.pdf"
    plt.savefig(outname)
    print outname, "ready"
    ax.clear()
