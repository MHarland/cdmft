import sys, numpy as np
from pytriqs.archive import HDFArchive
from pytriqs.gf.local import BlockGf, GfImTime, GfReFreq, GfImFreq
from pytriqs.utility import mpi

from cdmft.h5interface import Storage
from cdmft.gfoperations import trace


nambu = False
for archive_name in sys.argv[1:]:
    print "loading "+archive_name+"..."
    sto = Storage(archive_name)
    giw = sto.load("g_loc_iw")
    if nambu:
        for s, b in giw:
            for i in b.indices:
                i = int(i)
                if i%2:
                    b[i, i] << (-1) * b[i, i].conjugate()
    tr_giw = GfImFreq(indices = [0], mesh = giw.mesh)
    trace(giw, tr_giw)
    outname = archive_name[:-3]+'.txt'
    gdata = tr_giw.data[:,0,0]
    mesh = np.array([iw for iw in tr_giw.mesh])
    data = np.array([mesh.imag, gdata.real, gdata.imag]).T
    print data.shape
    np.savetxt(outname, data, header = 'Im(iwn) Re(G_loc) Im(G_loc)')
    print outname+' ready'
