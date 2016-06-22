from pytriqs.gf.local import BlockGf, GfImFreq, iOmega_n, inverse


class MatsubaraGreensFunction:
    
    def __init__(self, block_names, block_states, beta, n_iw):
        assert len(block_names) == len(block_states), "Number of Block-names and blocks have to equal"
        self.block_names = block_names
        self.block_states = block_states
        self.gf = BlockGf(name_list = block_names,
                          block_list = [GfImFreq(indices = states,
                                                 beta = beta,
                                                 n_points = n_iw) for states in block_states])
