import sys
from pytriqs.archive import HDFArchive
from pytriqs.gf.local import BlockGf, GfImTime, inverse
from pytriqs.utility import mpi
from maxent.bryanToTRIQS import MaximumEntropy

from cdmft.h5interface import Storage


ntau = 500
nomega = 1000
bandwidth = 15
sigma = 0.003 #0.003
nambu = True
groupname = 'maxent_orbs_results'
par = {"ntau": ntau,
       "nomega": nomega,
       "bandwidth": bandwidth,
       "sigma": sigma}
for archive_name in sys.argv[1:]:
    print archive_name
    sto = Storage(archive_name)
    #g = sto.load("g_tau")
    #"""
    gw = sto.load("g_imp_iw")
    """
    se = sto.load("se_imp_iw")
    for bn, b in se:
        b[0,1] << 0.
        b[1,0] << 0.
        b[2,3] << 0.
        b[3,2] << 0.
    g0 = sto.load("g0_iw")
    for bn, b in g0:
        b[0,1] << 0.
        b[1,0] << 0.
        b[2,3] << 0.
        b[3,2] << 0.
    gw << inverse(inverse(g0)-se)
    """
    if nambu:
        for s, b in gw:
            for i in b.indices:
                i = int(i)
                if i%2:
                    b[i, i] << (-1) * b[i, i].conjugate()
    g = BlockGf(name_block_generator = [(s, GfImTime(indices = [i for i in b.indices], beta = gw.beta, n_points = len(gw.mesh)*2))for s, b in gw])
    for s, b in gw:
        g[s].set_from_inverse_fourier(b)
    #"""
    #g['up-d-G'] << .5*(g['up-d-G'] + g['up-d-X'])
    #g['up-c-G'] << .5*(g['up-c-G'] + g['up-c-X'])
    aw = {}
    maxent = MaximumEntropy(g, ntau)
    for ind in g.all_indices:
        ind = (ind[0], int(ind[1]), int(ind[2]))
        b, i, j = ind
        if ind[1] != ind[2]: continue
        if i == 1 or i == 3: continue
        if b == 'XY' and i == 2: continue
        if 'dn' in ind[0]: continue
        #if 'GM' in ind[0]: continue
        #if 'X' in ind[0]: continue
        #if not('X' in ind[0]) and not('Y' in ind[0]): continue
        #if ind[1] in [1,3]: continue
        #if ind[1] == 1: continue
        print ind[0]+str(ind[1])+str(ind[2])
        if sigma:
            maxent.calculateDOS(nomega, bandwidth, sigma, orbital = ind)
        else:
            maxent.calculateDOS(nomega, bandwidth, orbital = ind)
        w = maxent.getOmegaMesh()
        aw[ind[0]+str(ind[1])+str(ind[2])] = maxent.getDOS()
    if mpi.is_master_node():
        arch = HDFArchive(archive_name, 'a')
        if not arch.is_group(groupname):
            arch.create_group(groupname)
            arch[groupname].create_group('a_w')
        res = arch[groupname]
        res['mesh'] = w
        ress = res['a_w']
        for k, v in aw.items():
            ress[k] = v
        #res['parameters'] = par
