import numpy as np
from bethe.parameters import DefaultDMFTParameters
from bethe.dmft import DMFT
from bethe.models import SingleSite as Model
from bethe.storage import LoopStorage


mod = Model(10, 0, 2, 1)
par = {"n_iw": 1025,
       "n_tau": 10001,
       "mix": 1,
       "make_g0_tau_real": False,
       "filling": 1,
       "dmu_max": .01,
       # solver:
       "n_cycles": 500000,
       #"partition_method": "autopartition",
       "partition_method": "quantum_numbers",
       "use_norm_as_weight": False,
       "measure_density_matrix": False,
       "length_cycle": 15,
       "n_warmup_cycles": 10000,
       "max_time": -1,
       "verbosity": 3,
       "move_shift": True,
       "move_double": True,
       "measure_g_tau": False,
       "measure_g_l": True,
       "measure_pert_order": False,
       "n_l": 25,
       # uses solver's fitting to the self-energy
       "perform_post_proc": True,
       "perform_tail_fit": True,
       "fit_min_w": 20,
       "fit_max_w": 40,
       "fit_max_moment": 3}
par = DefaultDMFTParameters(par, mod)
sto = LoopStorage("example.h5")
mod.set_initial_guess(**sto.provide_initial_guess())
dmft = DMFT(sto, par, **mod.init_dmft())
dmft.run_loops(4)
#dmft.mu = {"up": 1 * np.identity(1), "dn": 1 * np.identity(1)}
#dmft.run_loops(4, filling = 1)
