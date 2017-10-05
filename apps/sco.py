import matplotlib, sys, numpy as np, itertools as itt

from bethe.h5interface import Storage
from bethe.plot.cfg import plt, ax


for fname in sys.argv[1:]:
    print 'loading '+fname+'...'
    sto = Storage(fname)
    n_loops = sto.get_completed_loops()
    loops = range(n_loops)
    orbs = [(k,i,i) for k, i in itt.product(['G', 'M'], [0, 1])] + [(k,i,j) for k, i, j in itt.product(['X', 'Y'], [0, 1], [0, 1])]
    orders = ['ntot', 'sco', 'afm']
    nc = len(orbs)
    colors = [matplotlib.cm.jet(i/float(max(1,nc-1))) for i in range(nc)]
    graphs = {orb: [] for orb in orbs + orders}
    for l in loops:
        g = sto.load('g_imp_iw', l)
        for orb in orbs:
            ntot = 0
            b, i, j = orb
            if (i, j) == (1, 1):
                n = -1 * g[b][i, j].conjugate().total_density()
            else:
                n = g[b][i, j].total_density()
            if n.imag > 0.001:
                print 'warning: non-vanishing imaginary part!'
            n = n.real
            ntot += n
            graphs[orb].append(n)
        sco = (g['Y'][0, 1].total_density() + g['Y'][1, 0].total_density() - g['X'][0, 1].total_density() - g['X'][1, 0].total_density()) * .25
        afm = np.sum([g[k][0, 0].total_density() - (-1) * g[k][1, 1].conjugate().total_density() for k in ['G','X','Y','M']]) *.5
        for key, val in {'ntot': ntot, 'sco': sco, 'afm': afm}.items():
            graphs[key].append(val.real)
        print 'ntot('+str(l)+') = '+str(ntot.real)
        print 'sco('+str(l)+') = '+str(sco.real)
        print 'afm('+str(l)+') = '+str(afm.real)
    
    for order in orders:
        kwargs = {'label': "$\\mathrm{"+order+"}$"}
        ax.plot(loops, graphs[order], **kwargs)
    ax.legend(loc = "best", fontsize = 6)
    ax.set_xlabel("$\mathrm{DMFT-Loop}$")
    ax.set_ylabel("$<\\mathrm{orderparameter}$>")
    outname = fname[:-3]+"_sco1.pdf"
    plt.savefig(outname)
    print outname, "ready"
    ax.clear()
    
    for orb, c in zip(orbs, colors):
        b, i, j = orb
        kwargs = {'label': '$\\mathrm{'+b+'}'+str(i)+str(j)+'$', 'color': c}
        if i != j: kwargs['ls'] = '--'
        ax.plot(loops, graphs[orb], **kwargs)
    ax.legend(loc = "best", fontsize = 6)
    ax.set_xlabel("$\mathrm{DMFT-Loop}$")
    ax.set_ylabel("$n$")
    outname = fname[:-3]+"_sco2.pdf"
    plt.savefig(outname)
    print outname, "ready"
    ax.clear()
