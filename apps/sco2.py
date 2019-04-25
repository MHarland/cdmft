import matplotlib, sys, numpy as np, itertools as itt

from bethe.h5interface import Storage
from bethe.plot.cfg import plt, ax
from bethe.transformation import MatrixTransformation


verbose = False
if '-v' in sys.argv:
    verbose = True
    sys.argv.remove('-v')

gf_struct = [[k, range(4)] for k in ["GM", "XY"]]
gf_struct_site = [[s, range(4)] for s in ["up", "dn"]]
reblock_map = {("GM",0,0):("up",0,0), ("GM",1,1):("dn",0,0),
               ("GM",2,2):("up",1,1), ("GM",3,3):("dn",1,1),
               ("XY",0,0):("up",2,2), ("XY",1,1):("dn",2,2),
               ("XY",2,2):("up",3,3), ("XY",3,3):("dn",3,3),
               ("GM",0,2):("up",0,1), ("GM",2,0):("up",1,0),
               ("GM",1,3):("dn",0,1), ("GM",3,1):("dn",1,0),
               ("XY",0,2):("up",2,3), ("XY",2,0):("up",3,2),
               ("XY",1,3):("dn",2,3), ("XY",3,1):("dn",3,2)
}
transformation_matrix = .5 * np.array([[1,1,1,1],
                                       [1,-1,-1,1],
                                       [1,-1,1,-1],
                                       [1,1,-1,-1]]) # g m x y
transformation = dict([(s, transformation_matrix) for s in ["up", "dn"]])
reblock = MatrixTransformation(gf_struct, None, gf_struct_site, reblock_map)
transform = MatrixTransformation(gf_struct_site, transformation, gf_struct_site)

for fname in sys.argv[1:]:
    print 'loading '+fname+'...'
    sto = Storage(fname)
    n_loops = sto.get_completed_loops()
    loops = range(n_loops)
    orbs = [(k,i,j) for k, i, j in itt.product(['GM', 'XY'], range(4), range(4))]
    orders = ['N', 'SC', 'AFM', 'FM']
    nc = len(orbs)
    colors = [matplotlib.cm.jet(i/float(max(1,nc-1))) for i in range(nc)]
    graphs = {orb: [] for orb in orbs + orders}
    for l in loops:
        g = sto.load('g_imp_iw', l)
        ntot = 0
        for orb in orbs:
            b, i, j = orb
            if (i, j) in [(1, 1), (1, 3), (3, 1), (3, 3)]:
                n = -1 * g[b][i, j].conjugate().total_density()
            else:
                n = g[b][i, j].total_density()
            if n.imag > 0.001:
                print 'warning: non-vanishing imaginary part in',b,i,j
            n = n.real
            graphs[orb].append(n)
            if i == j:
                ntot += n
        sco = (g['XY'][0, 1].total_density() + g['XY'][1, 0].total_density() - g['XY'][2, 3].total_density() - g['XY'][3, 2].total_density()) * .25
        #print g['XY'][0, 1].total_density(), g['XY'][1, 0].total_density(), g['XY'][2, 3].total_density(), g['XY'][3, 2].total_density()
        gsite = reblock.reblock_by_map(g)
        gsite = transform.transform_g(gsite, False)
        gsite["dn"] << -1 * gsite["dn"].conjugate()
        afm = .125* (gsite["up"][0, 0] +gsite["up"][3, 3] +gsite["dn"][1, 1] +gsite["dn"][2, 2]
                     -gsite["dn"][0, 0] -gsite["dn"][3, 3] -gsite["up"][1, 1] -gsite["up"][2, 2]).total_density()
        fm = .125* (gsite["up"][0, 0] +gsite["up"][3, 3] +gsite["up"][1, 1] +gsite["up"][2, 2]
                     -gsite["dn"][0, 0] -gsite["dn"][3, 3] -gsite["dn"][1, 1] -gsite["dn"][2, 2]).total_density()

        for key, val in {'N': ntot, 'SC': sco, 'AFM': afm, 'FM': fm}.items():
            graphs[key].append(abs(val.real))
        if verbose and l == loops[-1]:
            print 'N('+str(l)+') = '+str(np.round(ntot.real, 3))
            print 'delta('+str(l)+') = '+str(np.round(1-ntot.real*.25, 3))
            print 'SC('+str(l)+') = '+str(np.round(sco.real, 3))
            print 'AFM('+str(l)+') = '+str(np.round(afm.real, 3))
            print 'FM('+str(l)+') = '+str(np.round(fm.real, 3))
            print g['XY'][0, 1].total_density().real, g['XY'][1, 0].total_density().real, g['XY'][2, 3].total_density().real, g['XY'][3, 2].total_density().real
            print gsite["up"][0, 0].total_density().real, gsite["up"][3, 3].total_density().real, gsite["dn"][1, 1].total_density().real, gsite["dn"][2, 2].total_density().real
            print gsite["dn"][0, 0].total_density().real, gsite["dn"][3, 3].total_density().real, gsite["up"][1, 1].total_density().real, gsite["up"][2, 2].total_density().real
    
    for order in orders:
        kwargs = {'label': "$\\mathrm{"+order+"}$"}
        ax.semilogy(loops, graphs[order], **kwargs)
    ax.set_ylim(bottom = 10**(-5))
    ax.legend(loc = "best", fontsize = 6, title = "$O$")
    ax.set_xlabel("$\mathrm{DMFT-Loop}$")
    ax.set_ylabel("$<O>$")
    outname = fname[:-3]+"_sco1.pdf"
    plt.savefig(outname)
    print outname, "ready"
    ax.clear()
    
    for orb, c in zip(orbs, colors):
        b, i, j = orb
        kwargs = {'label': '$\\mathrm{'+b+'}'+str(i)+str(j)+'$', 'color': c}
        if i != j: kwargs['ls'] = '--'
        ax.plot(loops, graphs[orb], **kwargs)
    ax.legend(loc = "upper left", fontsize = 6, framealpha = .5)
    ax.set_xlabel("$\mathrm{DMFT-Loop}$")
    ax.set_ylabel("$n$")
    outname = fname[:-3]+"_sco2.pdf"
    plt.savefig(outname)
    print outname, "ready"
    ax.clear()

    ax.plot(loops, graphs['SC'])
    ax.set_xlabel("$\mathrm{DMFT-Loop}$")
    ax.set_ylabel("$SC$")
    outname = fname[:-3]+"_sco3.pdf"
    plt.savefig(outname)
    print outname, "ready"
    ax.clear()
