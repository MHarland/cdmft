import numpy as np

from cdmft.setups.generic import CycleSetupGeneric
from cdmft.operators.hubbard import Site
from cdmft.schemes.dmft import GLocal, WeissField, SelfEnergy
from cdmft.tightbinding import LatticeDispersion


class SingleSite(CycleSetupGeneric):
    def __init__(self, beta, mu, u, t = -1, nk = 64, n_iw = 1025):
        up = "up"
        dn = "dn"
        spins = [up, dn]
        sites = range(1)
        hubbard = Site(u)
        blocknames = spins
        blocksizes = [len(sites), len(sites)]
        gf_struct = [[s, sites] for s in spins]
        hopping = {(1,0): [[t]],
                   (0,1): [[t]],
                   (-1,0): [[t]],
                   (0,-1): [[t]]
        }
        disp = LatticeDispersion(hopping, nk)
        self.ops = hubbard
        self.h_int = hubbard
        self.gloc = GLocal(disp, blocknames, blocksizes, beta, n_iw)
        self.g0 = WeissField(blocknames, blocksizes, beta, n_iw)
        self.se = SelfEnergy(blocknames, blocksizes, beta, n_iw)
        self.mu = mu
        self.global_moves = {"spin-flip": {("up", 0): ("dn", 0), ("dn", 0): ("up", 0)}}
        self.quantum_numbers = [hubbard.get_n_tot(), hubbard.get_n_per_spin(up)]
