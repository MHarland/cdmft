import sys
from cdmft.h5interface import Storage
from cdmft.convergence import Criterion


for fname in sys.argv[1:]:
    sto = Storage(fname)
    crit = Criterion(sto)
    print fname+':',crit.confirms_convergence()

