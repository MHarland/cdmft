import itertools as itt, numpy as np

from pytriqs.gf.local import BlockGf, GfImFreq, inverse, GfImTime, TailGf


class MatsubaraGreensFunction(BlockGf):
    """
    Greens functions interface to TRIQS. Provides convenient initialization.
    gf_init creates a MGF with the same structure as gf_init, but no value initialization
    __lshift__, __isub__, __iadd__ had to be extended in order to make them available to childs of
    MatsubaraGreensFunction
    """
    def __lshift__(self, x):
        if isinstance(x, MatsubaraGreensFunction) or isinstance(x, BlockGf):
            for i, g in self: g.copy_from(x[i])
            return self
        else:
            BlockGf.__lshift__(self, x)
    
    def __isub__(self, x):
        if isinstance(x, MatsubaraGreensFunction) or isinstance(x, BlockGf):
            for (n,g) in self:
                self[n] -= x[n]
        else:
            g = self.get_as_BlockGf()
            g -= x
            self << g
        return self
    
    def __iadd__(self, x):
        if isinstance(x, MatsubaraGreensFunction) or isinstance(x, BlockGf):
            for (n,g) in self:
                self[n] += x[n]
        else:
            g = self.get_as_BlockGf()
            g += x
            self << g
        return self

    def copy(self):
        #g = self.__class__(gf_init = self)
        g = self.get_as_BlockGf()
        g << self
        return g

    def __init__(self, blocknames = None, blocksizes = None, beta = None, n_iw = None, name = '', gf_init = None, verbosity = 0, **kwargs):
        kwargskeys = [k for k in kwargs.keys()]
        if type(gf_init) == BlockGf:
            blocknames = [i for i in gf_init.indices]
            blocksizes = [len([i for i in b.indices]) for bn, b in gf_init]
            beta = gf_init.beta
            n_iw = int(len(gf_init.mesh) * .5)
            BlockGf.__init__(self, name_block_generator = [(bn, GfImFreq(beta = beta, n_points = n_iw, indices = range(bs))) for bn, bs in zip(blocknames, blocksizes)], name = name)
        elif isinstance(gf_init, MatsubaraGreensFunction):
            assert isinstance(gf_init, MatsubaraGreensFunction), "gf_init must be a Matsubara GreensFunction"
            blocknames = gf_init.blocknames
            blocksizes = gf_init.blocksizes
            beta = gf_init.beta
            n_iw = gf_init.n_iw
            BlockGf.__init__(self, name_block_generator = [(bn, GfImFreq(beta = beta, n_points = n_iw, indices = range(bs))) for bn, bs in zip(blocknames, blocksizes)], name = name)
        elif 'name_block_generator' in kwargskeys: # TODO test
            blocknames = [block[0] for block in kwargs['name_block_generator'].values()]
            blocksizes = [block[1].N1 for block in kwargs['name_block_generator'].values()]
            beta = kwargs['name_block_generator'][0][1].beta
            n_iw = int(len(kwargs['name_block_generator'][0][1].mesh) * .5)
            BlockGf.__init__(self, **kwargs)
        elif 'name_list' in kwargskeys: # TODO test
            blocknames = kwargs['name_list']
            blocksizes = [g.N1 for g in kwargs['block_list']]
            beta = kwargs['block_list'][0].beta
            n_iw = int(len(kwargs['block_list'][0].mesh) * .5)
            BlockGf.__init__(self, **kwargs)
        else:
            assert blocknames is not None and blocksizes is not None and beta is not None and n_iw is not None, "Missing parameter for initialization without gf_init"
            assert len(blocknames) == len(blocksizes), "Number of Block-names and blocks have to equal"
            BlockGf.__init__(self, name_block_generator = [(bn, GfImFreq(beta = beta, n_points = n_iw, indices = range(bs))) for bn, bs in zip(blocknames, blocksizes)], name = name)
        self.blocknames = blocknames
        self.blocksizes = blocksizes
        self.n_iw = n_iw
        self.gf_struct = [(bn, range(bs)) for bn, bs in zip(blocknames, blocksizes)]
        self._gf_lastloop = None
        self.verbosity = verbosity

    def prepare_mix(self):
        self._gf_lastloop = self.copy()

    def mix(self, coeff):
        """
        mixes with the solution of the previous loop, coeff is the weight of the new state
        """
        if not coeff is None:
            self << coeff * self + (1 - coeff) * self._gf_lastloop
            self._gf_lastloop << self

    def symmetrize(self, block_symmetries):
        """
        imposes symmetries, each sublist of block_symmetries represents a symmetry-class
        """
        for symmetry in block_symmetries:
            self._symmetrize_block(symmetry)

    def _symmetrize_block(self, symmetry):
        for s1, b1 in self:
            for blocklabel_sym_part in symmetry:
                if blocklabel_sym_part in s1:
                    sublabel = s1.replace(blocklabel_sym_part, "")
                    for s2, b2 in self:
                        symlabel_in_s2 = False
                        for sym in symmetry:
                            if sym in s2:
                                symlabel_in_s2 = True
                        if sublabel in s2 and s1 != s2 and symlabel_in_s2:
                            b1 << .5 * (b1 + b2)
                            b2 << b1

    def get_as_BlockGf(self):
        """
        returns object as BlockGf, e.g. for writing it into HDFArchives. That process is only
        defined for the parent class BlockGf.
        """
        g = BlockGf(name_block_generator = [(bn, GfImFreq(beta = self.beta, n_points = self.n_iw, indices = range(bs))) for bn, bs in zip(self.blocknames, self.blocksizes)], name = self.name)
        g << self
        return g

    def make_g_tau_real(self, n_tau):
        """
        Transforms to tau space with n_tau meshsize, sets self accordingly
        TODO tail
        """
        assert False, "something might be wrong here" # TODO
        inds_tau = range(n_tau)
        g_tau = BlockGf(name_list = self.blocknames,
                         block_list = [GfImTime(beta = self.beta, indices = range(s),
                                                n_points = n_tau) for s in self.blocksizes])
        for bname, b in g_tau:
            b.set_from_inverse_fourier(self[bname])
            inds_block = range(len(b.data[0,:,:]))
            for n, i, j in itt.product(inds_tau, inds_block, inds_block):
                b.data[n,i,j] = b.data[n,i,j].real
            self[bname].set_from_fourier(b)

    def fit_tail2(self, w_start_fit, w_stop_fit, max_mom_to_fit = 3, known_moments = [(1, np.identity(1))]):
        for s, b in self:
            n1 = int(self.beta/(2*np.pi)*w_start_fit -.5)
            n2 = int(self.beta/(2*np.pi)*w_stop_fit -.5)
            tails = TailGf(b.N1, b.N2)
            for moment in known_moments:
                tails[moment[0]] = moment[1]
            b.fit_tail(tails, max_mom_to_fit, n1, n2)

    def _to_blockmatrix(self, number):
        bmat = dict()
        for bname, bsize in zip(self.blocknames, self.blocksizes):
            bmat[bname] = np.identity(bsize) * number
        return bmat

    def _quickplot(self, file_name, x_range = (0, 100)):
        """
        for debugging
        """
        from matplotlib import pyplot as plt
        mesh = np.array([w.imag for w in self.mesh])
        ia, ie = x_range[0] + self.n_iw, x_range[1] + self.n_iw
        for s, b in self:
            orbs = range(b.data.shape[1])
            for i, j in itt.product(orbs, orbs):
                plt.plot(mesh[ia:ie], b.data[ia:ie, i, j].imag)
                plt.plot(mesh[ia:ie], b.data[ia:ie, i, j].real, ls = 'dashed')
        plt.savefig(file_name)
        plt.close()
