from cdmft.storage import LoopStorage
from cdmft.evaluation.densitymatrix import StaticObservable
from cdmft.hamiltonian import Site

sto = LoopStorage("example.h5")
ss = Site(1, ["up", "dn"])
ss_static = StaticObservable(ss.ss(0,0), sto)
print ss_static.get_expectation_value()
