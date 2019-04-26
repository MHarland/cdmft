import sys, itertools
from pytriqs.archive import HDFArchive
from pytriqs.gf.local import BlockGf, GfImTime, inverse
from pytriqs.utility import mpi
from maxent.bryanToTRIQS import MaximumEntropy

from cdmft.h5interface import Storage


ntau = 1000
nomega = 2000
bandwidth = 15#10#20#30
#sigma = 0.0003
sigma = 0.003
nambu = False
hfl = False
gn = 'maxent_results'
if hfl:
    gn += '_hfl'
par = {"ntau": ntau,
       "nomega": nomega,
       "bandwidth": bandwidth,
       "sigma": sigma}

for archive_name in sys.argv[1:]:
    print archive_name
    sto = Storage(archive_name)
    #g = sto.load("g_tau")
    gw = sto.load("g_imp_iw")
    if hfl:
        #zeros = [(0,2),(0,3),(1,2),(1,3)]#dsc
        #zeros = [(0,1),(0,3),(1,2),(2,3)]#afm
        #zeros = [(0,1),(0,2),(1,3),(2,3)]#triplet-dsc
        zeros = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]#paramag
        #zeros = [(0,2),(1,3)]#kill afm
        se = sto.load("se_imp_iw")
        for bn, b in se:
            for i,j in zeros:
                b[i,j] << 0.
                b[j,i] << 0.
        g0 = sto.load("g0_iw")
        for bn, b in g0:
            for i,j in itertools.product(range(4), range(4)):
                if i==j:continue
                b[i,j] << 0.
        gw << inverse(inverse(g0)-se)
    #gw = sto.load("g_loc_iw") # !!!!!!!!!!!!!!!!!!!
    if nambu:
        for s, b in gw:
            for i in b.indices:
                i = int(i)
                if i%2:
                    b[i, i] << (-1) * b[i, i].conjugate()
    g = BlockGf(name_block_generator = [(s, GfImTime(indices = [i for i in b.indices], beta = gw.beta, n_points = len(gw.mesh)*2))for s, b in gw])
    for s, b in gw:
        g[s].set_from_inverse_fourier(b)


    maxent = MaximumEntropy(g, ntau)
    if sigma:
        maxent.calculateTotDOS(nomega, bandwidth, sigma)
    else:
        maxent.calculateTotDOS(nomega, bandwidth)
    w = maxent.getOmegaMesh()
    a = maxent.getTotDOS()
    if mpi.is_master_node():
        arch = HDFArchive(archive_name, 'a')
        if not arch.is_group(gn):
            arch.create_group(gn)
        res = arch[gn]
        res['mesh'] = w
        res['a_w'] = a
        #res['parameters'] = par
