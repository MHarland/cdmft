from cdmft.h5interface import Storage
from cdmft.selfconsistency import Cycle
from cdmft.setups.cdmftsquarelattice import (
    MomentumPlaquetteSetup, NambuMomentumPlaquetteSetup
)
from cdmft.parameters import DefaultDMFTParameters


# mu = U/2 is often a good start as it is half-filling
# for bipartite lattices. Here mu = U/2 - tnnn
setup_normal = MomentumPlaquetteSetup(25, 3.7, 8, -1, 0.3, 32)
sto = Storage('ex_cdmft.h5')
par = DefaultDMFTParameters()
par['n_l'] = 50
par['measure_G_l'] = True
par['measure_G_tau'] = False
par['measure_pert_order'] = True
par['length_cycle'] = 25 # better: >= 50
par['n_cycles'] = 100000 # better 10**7
par['n_warmup_cycles'] = 5000 # 1% to 5% of n_cycles
cyc = Cycle(sto, par, **setup_normal.initialize_cycle())
cyc.run(1)
# mu will be fitted from here on, it requires a "good" self-energy
# which is why I recommend to set filling after one or two dmft cycles
cyc.g_loc.filling = 3.8
cyc.run(1)

# mu will be loaded, I expect convergence to be more stable with constant mu
setup = NambuMomentumPlaquetteSetup(25, 999, 8, -1, 0.3, 32)
setup.set_data(sto, load_mu=True)
sto = Storage('ex_cdmft_nambu.h5')
par = DefaultDMFTParameters()
par['n_l'] = 50
par['measure_G_l'] = True
par['measure_G_tau'] = False
par['measure_pert_order'] = True
par['length_cycle'] = 25
par['n_cycles'] = 100000
par['n_warmup_cycles'] = 5000
setup.apply_dynamical_sc_field(0.05)
cyc = Cycle(sto, par, **setup.initialize_cycle())
cyc.run(1)
