import sys, numpy as np, itertools

from bethe.h5interface import Storage
from bethe.transformation import MatrixTransformation

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
    print fname+':'
    sto = Storage(fname)
    g = sto.load('g_imp_iw')
    gsite = reblock.reblock_by_map(g)
    gsite = transform.transform_g(gsite, False)
    for s, b in gsite:
        print s
        dens = np.zeros(b.data.shape[1:])
        inds = range(len(dens))
        for i,j in itertools.product(inds, inds):
            dens[i,j] = b[i,j].density().real
        print np.round(dens,4)
