import numpy as np

from bethe.setups.generic import CycleSetupGeneric
from bethe.operators.hubbard import Site
from bethe.schemes.hypercubic import GLocal, WeissField, SelfEnergy


class HypercubicSetup(CycleSetupGeneric):
    """
    """
    def __init__(self, beta, mu, u, t = 1, rho_wmin = -20, rho_wmax = 20, rho_npts = 4000, w1 = None, w2 = None, n_mom = 3, n_iw = 1025):
        up = "up"
        dn = "dn"
        spins = [up, dn]
        sites = range(1)
        hubbard = Site(u)
        blocknames = spins
        blocksizes = [len(sites), len(sites)]
        gf_struct = [[s, sites] for s in spins]
        self.h_int = hubbard
        self.gloc = GLocal(t, rho_wmin, rho_wmax, rho_npts, w1, w2, n_mom, blocknames, blocksizes, beta, n_iw)
        self.g0 = WeissField(blocknames, blocksizes, beta, n_iw)
        self.se = SelfEnergy(blocknames, blocksizes, beta, n_iw)
        self.mu = mu
        self.global_moves = {"spin-flip": {("up", 0): ("dn", 0), ("dn", 0): ("up", 0)}}
        self.quantum_numbers = [hubbard.get_n_tot()]#, hubbard.get_n_per_spin(up)]
