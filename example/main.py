from bethe.parameters import DefaultDMFTParameters
from bethe.dmft import DMFT
from bethe.models import SingleSiteBethe


mod = SingleSiteBethe(10, 1, 2)
par = {"n_iw": 1025,
       "n_tau": 10001,
       # solver:
       "n_cycles": 100000,
       "partition_method": "autopartition",
       "length_cycle": 15,
       "n_warmup_cycles": 10000,
       "max_time": -1,
       "verbosity": 1,
       "move_shift": True,
       "move_double": True,
       "measure_g_tau": True,
       # uses solver's fitting to the self-energy
       "perform_post_proc": True,
       "perform_tail_fit": True,
       "fit_min_w": 10,
       "fit_max_w": 20,
       "fit_max_moment": 3}
par = DefaultDMFTParameters(par)
dmft = DMFT(par, mod, "example.h5")
dmft.run_loops(2)
