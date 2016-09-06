from pytriqs.gf.local import BlockGf, GfImFreq, iOmega_n, inverse


class MatsubaraGreensFunction:
    """
    TRIQS BlockGf is found in self.gf. This class extends it by some special features.
    Inheritance has been avoided, due to some complex features provided by TRIQS.
    """
    def __init__(self, block_names, block_states, beta, n_iw, gf_init = None, *args, **kwargs):
        assert len(block_names) == len(block_states), "Number of Block-names and blocks have to equal"
        self.block_names = block_names
        self.block_states = block_states
        self.beta = beta
        self.n_iw = n_iw
        self.gf = BlockGf(name_list = block_names,
                          block_list = [GfImFreq(indices = states,
                                                 beta = beta,
                                                 n_points = n_iw) for states in block_states])
        self._gf_lastloop = self.gf.copy() 
        if not gf_init is None:
            self.gf << g_init
            self._gf_lastloop << g_init

    def set_gf(self, *args):
        """sets the first non-None argument, dropping the remainers"""
        for gf in list(args):
            assert isinstance(gf, BlockGf) or isinstance(gf, MatsubaraGreensFunction), str(type(gf))+" not recognized"
            if isinstance(gf, BlockGf):
                self.gf << gf
                break
            elif not gf is None:
                self.gf << self.gf
                break

    def mix(self, coeff):
        """mixes with the solution of the previous loop, coeff is the weight of the 
        new state"""
        if not coeff is None:
            self.gf << coeff * self.gf + (1 - coeff) * self._gf_lastloop
            self._gf_lastloop << self.gf

    def symmetrize(self, block_symmetries):
        """imposes symmetries each sublist in block_symmetries represents a 
        symmetry-class"""
        for symmetry in block_symmetries:
            self._symmetrize_block(symmetry)

    def _symmetrize_block(self, symmetry):
        for s1, b1 in self.gf:
            for blocklabel_sym_part in symmetry:
                if blocklabel_sym_part in s1:
                    sublabel = s1.replace(blocklabel_sym_part, "")
                    for s2, b2 in self.gf:
                        symlabel_in_s2 = False
                        for sym in symmetry:
                            if sym in s2:
                                symlabel_in_s2 = True
                        if sublabel in s2 and s1 != s2 and symlabel_in_s2:
                            b1 << .5 * (b1 + b2)
                            b2 << b1
