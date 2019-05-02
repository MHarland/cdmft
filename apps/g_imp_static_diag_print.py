import sys, numpy as np

from cdmft.evaluation.common import Evaluation
from cdmft.h5interface import Storage


for fname in sys.argv[1:]:
    sto = Storage(fname)
    ev = Evaluation(sto)
    print
    print fname+':'
    for orbname, nstatic in ev.get_g_static_blockdiags().items():
        print orbname+':', np.round(nstatic, 2)
