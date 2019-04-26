from cdmft.h5interface import Storage
from cdmft.selfconsistency import Cycle
from cdmft.setups.cdmftsquarelattice import MomentumPlaquetteSetup
from cdmft.parameters import DefaultDMFTParameters


setup = MomentumPlaquetteSetup(10, 2, 4, -1, 0, 16)
sto = Storage('ex_cdmft.h5')
par = DefaultDMFTParameters()
par['n_l'] = 35
par['measure_g_l'] = True
par['measure_g_tau'] = False
par['measure_pert_order'] = True
cyc = Cycle(sto, par, **setup.initialize_cycle())
cyc.run(5, n_cycles = int(2.5*10**5), filling = None)

