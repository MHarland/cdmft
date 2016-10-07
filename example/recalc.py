from bethe.parameters import DefaultDMFTParameters
from bethe.dmft import DMFT
from bethe.models import SingleSite as Model
from bethe.storage import LoopStorage


mod = Model(10, 1, 2, 1)
par = {"n_iw": 1025,
       "n_tau": 10001,
       "mix": 1,
       "make_g0_tau_real": False,
       "filling": None,
       # solver:
       "n_cycles": 100000,
       "partition_method": "autopartition",
       "length_cycle": 15,
       "n_warmup_cycles": 10000,
       "max_time": -1,
       "verbosity": 2,
       "move_shift": True,
       "move_double": True,
       "measure_g_tau": False,
       "measure_g_l": True,
       "n_l": 30,
       # uses solver's fitting to the self-energy
       "perform_post_proc": True,
       "perform_tail_fit": True,
       "fit_min_w": 20,
       "fit_max_w": 40,
       "fit_max_moment": 3}
par = DefaultDMFTParameters(mod, par)
sto = LoopStorage("example.h5")
mod.set_initial_guess(**sto.provide_initial_guess())
dmft = DMFT(sto, par, **mod.init_dmft())
dmft.run_loops(2)
