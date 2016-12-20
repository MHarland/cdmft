import numpy as np

from bethe.h5interface import Storage
from bethe.schemes.cdmft import SelfEnergy
from bethe.selfconsistency import Cycle
from bethe.setups.cdmftsquarelattice import MomentumPlaquetteSetup
from bethe.transformation import MatrixTransformation
from bethe.parameters import DefaultDMFTParameters


beta = 10
setup = MomentumPlaquetteSetup(beta, 2, 4, -1, 0, 16)
site_to_mom = MatrixTransformation(setup.old_struct, setup.site_transf_mat, setup.new_struct)
se_site = SelfEnergy(beta = beta, gf_struct = setup.old_struct)
se_site['up'].data[:,:,:] = np.zeros([se_site.n_iw*2, 4, 4]) #data here, direct access to the data must be treated carefully: there are n_iw * 2 data-entries, due to negative matsubara frequencies, and the consistency with the tail gets lost, see below
se_site['dn'].data[:,:,:] = np.zeros([se_site.n_iw*2, 4, 4]) #data here
#se_site.fit_tail2(20, 30, 3, []) #eventually, if high frequency data is non-analytic, i.e. noisy
setup.se << site_to_mom.transform_g(se_site) #transform and change blockstructure from 2x4 to 8x1
sto = Storage('ex_cdmft.h5')
par = DefaultDMFTParameters()
cyc = Cycle(sto, par, **setup.initialize_cycle())
cyc.run(4, n_cycles = 10**5, filling = None, perform_tail_fit = False) #if "need more arguments to fit tail" -error occurs you need more n_cycles to reduce noise
se_site << site_to_mom.backtransform_g(cyc.se)

