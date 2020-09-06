from cdmft.h5interface import Storage
from cdmft.selfconsistency import Cycle
from cdmft.setups.hypercubic import HypercubicSetup
from cdmft.parameters import DefaultDMFTParameters


setup = HypercubicSetup(5, 2, 4, -1)
sto = Storage('ex_hypercubic.h5')
par = DefaultDMFTParameters()
par['measure_G_tau'] = False
par['n_cycles'] = 10**5
cyc = Cycle(sto, par, **setup.initialize_cycle())
cyc.run(3)
