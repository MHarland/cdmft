from pytriqs.gf.local import BlockGf, GfImFreq
from pytriqs.utility.bound_and_bisect import bound_and_bisect

from greensfunctions import MatsubaraGreensFunction


class GLocal(MatsubaraGreensFunction):

    def __init__(self, name_list, block_states, beta, n_iw):
        MatsubaraGreensFunction.__init__(self, name_list, block_states, beta, n_iw)

    def set_mu(self):
        pass
