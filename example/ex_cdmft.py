from bethe.h5interface import Storage
from bethe.selfconsistency import Cycle
from bethe.setups.cdmftsquarelattice import MomentumPlaquetteSetup
from bethe.parameters import DefaultDMFTParameters


setup = MomentumPlaquetteSetup(10, 2, 4, -1, 0, 16)
sto = Storage('ex_cdmft.h5')
par = DefaultDMFTParameters()
cyc = Cycle(sto, par, **setup.initialize_cycle())
cyc.run(1, n_cycles = 10**5, perform_tail_fit = False, filling = 4)

