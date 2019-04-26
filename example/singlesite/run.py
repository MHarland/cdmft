from cdmft.setups.singlesite import SingleSite
from cdmft.selfconsistency import Cycle
from cdmft.parameters import DefaultDMFTParameters
from cdmft.h5interface import Storage
from scipy.constants import k, e

t = -1.
w = -8*t
u = 1.5*w
mu = u*.5
beta = .34*e/(-t*k*270)
assert 4 < beta < 301, "beta is "+str(beta)
print t,w,u,beta
sto = Storage('singlesite.h5')
setup = SingleSite(beta, mu, u, t = t, nk = 32)
par = DefaultDMFTParameters()
par['n_l'] = int(.5*beta)+35
par['cycle_length'] = 200
par['n_cycles'] = 10000
par['n_warmup_cycles'] = 1000
par['measure_g_l'] = True
par['mix'] = .3
#par['block_symmetries'] = [['up', 'dn']]
par['measure_g_tau'] = False
cyc = Cycle(sto, par, **setup.initialize_cycle())
cyc.se << mu
cyc.run(1)
cyc.h_int = cyc.h_int.get_h_int() + .1 * setup.ops.sz(0)
cyc.run(1)
cyc.h_int -= .1 * setup.ops.sz(0)
cyc.run(10)
