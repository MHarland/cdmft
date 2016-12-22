import sys

from bethe.evaluation.generic import Evaluation
from bethe.h5interface import Storage


for fname in sys.argv[1:]:
    sto = Storage(fname)
    ev = Evaluation(sto)
    print
    print fname+':'
    for orbname, nstatic in ev.get_g_static_diags().items():
        print orbname+':', nstatic
