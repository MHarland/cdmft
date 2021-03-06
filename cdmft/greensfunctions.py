import itertools as itt
import numpy as np

from pytriqs.gf import BlockGf, GfImFreq, inverse, GfImTime, make_zero_tail, replace_by_tail, fit_tail_on_window, fit_hermitian_tail_on_window


class MatsubaraGreensFunction(BlockGf):
    """
    Greens functions interface to TRIQS. Provides convenient initialization.
    gf_init creates a MGF with the same structure as gf_init, but no value initialization
    __lshift__, __isub__, __iadd__ had to be extended in order to make them available to childs of
    MatsubaraGreensFunction
    """

    def __lshift__(self, x):
        if isinstance(x, MatsubaraGreensFunction) or isinstance(x, BlockGf):
            for i, g in self:
                g.copy_from(x[i])
            return self
        else:
            BlockGf.__lshift__(self, x)

    def __isub__(self, x):
        if isinstance(x, MatsubaraGreensFunction) or isinstance(x, BlockGf):
            for (n, g) in self:
                self[n] -= x[n]
        else:
            g = self.get_as_BlockGf()
            g -= x
            self << g
        return self

    def __iadd__(self, x):
        if isinstance(x, MatsubaraGreensFunction) or isinstance(x, BlockGf):
            for (n, g) in self:
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

    @property
    def all_indices(self):
        inds = list()
        for bn, b in self:
            for i, j in itt.product(b.indices[0], b.indices[0]):
                inds.append((bn, int(i), int(j)))
        return inds

    def __init__(self, blocknames=None, blocksizes=None, beta=None, n_iw=1025, name='', gf_init=None, gf_struct=None, verbosity=0, **kwargs):
        kwargskeys = [k for k in kwargs.keys()]
        if type(gf_init) == BlockGf:
            blocknames = [i for i in gf_init.indices]
            blocksizes = [len([i for i in b.indices]) for bn, b in gf_init]
            beta = gf_init.mesh.beta
            n_iw = int(len(gf_init.mesh) * .5)
            super(MatsubaraGreensFunction, self).__init__(name_block_generator=[(bn, GfImFreq(
                beta=beta, n_points=n_iw, indices=range(bs))) for bn, bs in zip(blocknames, blocksizes)], name=name, make_copies=False)
        elif isinstance(gf_init, MatsubaraGreensFunction):
            assert isinstance(
                gf_init, MatsubaraGreensFunction), "gf_init must be a Matsubara GreensFunction"
            blocknames = gf_init.blocknames
            blocksizes = gf_init.blocksizes
            beta = gf_init.mesh.beta
            n_iw = gf_init.n_iw
            super(MatsubaraGreensFunction, self).__init__(name_block_generator=[(bn, GfImFreq(
                beta=beta, n_points=n_iw, indices=range(bs))) for bn, bs in zip(blocknames, blocksizes)], name=name, make_copies=False)
        elif 'name_block_generator' in kwargskeys:  # TODO test
            blocknames = [block[0]
                          for block in kwargs['name_block_generator']]
            blocksizes = [
                block[1].target_shape[0] for block in kwargs['name_block_generator']]
            beta = kwargs['name_block_generator'][0][1].mesh.beta
            n_iw = int(len(kwargs['name_block_generator'][0][1].mesh) * .5)
            super(MatsubaraGreensFunction, self).__init__(**kwargs)
        elif 'name_list' in kwargskeys:  # TODO test
            blocknames = kwargs['name_list']
            blocksizes = [g.target_shape[0] for g in kwargs['block_list']]
            beta = kwargs['block_list'][0].mesh.beta
            n_iw = int(len(kwargs['block_list'][0].mesh) * .5)
            super(MatsubaraGreensFunction, self).__init__(**kwargs)
        elif gf_struct is not None:
            assert type(
                gf_struct) == list, "gf_struct must be of list-type here"
            blocknames = [b[0] for b in gf_struct]
            blocksizes = [len(b[1]) for b in gf_struct]
            beta = beta
            n_iw = n_iw
            super(MatsubaraGreensFunction, self).__init__(name_block_generator=[(bn, GfImFreq(
                beta=beta, n_points=n_iw, indices=range(bs))) for bn, bs in zip(blocknames, blocksizes)], name=name, make_copies=False)
        else:
            assert blocknames is not None and blocksizes is not None and beta is not None and n_iw is not None, "Missing parameter for initialization without gf_init and gf_struct"
            assert len(blocknames) == len(
                blocksizes), "Number of Block-names and blocks have to equal"
            super(MatsubaraGreensFunction, self).__init__(name_block_generator=((bn, GfImFreq(
                beta=beta, n_points=n_iw, indices=range(bs))) for bn, bs in zip(blocknames, blocksizes)), name=name, make_copies=False)
        self.blocknames = blocknames
        self.blocksizes = blocksizes
        self.n_iw = n_iw
        self.iw_offset = int(.5 * self.n_iw)
        self.gf_struct = [(bn, range(bs))
                          for bn, bs in zip(blocknames, blocksizes)]
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
        g = BlockGf(name_block_generator=((bn, GfImFreq(beta=self.mesh.beta, n_points=self.n_iw, indices=range(
            bs))) for bn, bs in zip(self.blocknames, self.blocksizes)), name=self.name, make_copies=False)
        g << self
        return g

    def make_g_tau_real(self, n_tau):
        """
        Transforms to tau space with n_tau meshsize, sets self accordingly
        TODO tail
        """
        self.fit_tail2()
        inds_tau = range(n_tau)
        g_tau = BlockGf(name_list=self.blocknames,
                        block_list=[GfImTime(beta=self.mesh.beta, indices=range(s),
                                             n_points=n_tau) for s in self.blocksizes])
        for bname, b in g_tau:
            b.set_from_fourier(self[bname])
            inds_block = range(len(b.data[0, :, :]))
            for n, i, j in itt.product(inds_tau, inds_block, inds_block):
                b.data[n, i, j] = b.data[n, i, j].real
            self[bname].set_from_fourier(b)

    def fit_tail2(self, known_moments=None, hermitian=True, fit_min_n=None, fit_max_n=None, fit_min_w=None, fit_max_w=None, fit_max_moment=None):
        """
        (simplified) interface to TRIQS fit_ for convenience.
        TRIQS fit_tail is also directly available
        """
        if fit_min_w is not None:
            fit_min_n = int(0.5*(fit_min_w*self.mesh.beta/np.pi - 1.0))
        if fit_max_w is not None:
            fit_max_n = int(0.5*(fit_max_w*self.mesh.beta/np.pi - 1.0))
        if fit_min_n is None:
            fit_min_n = int(0.8*len(self.mesh)/2)
        if fit_max_n is None:
            fit_max_n = int(len(self.mesh)/2)
        if fit_max_moment is None:
            fit_max_moment = 3
        for bn, b in self:
            if known_moments is None:
                known_moments = make_zero_tail(b, 2)
                known_moments[1] = np.eye(b.target_shape[0])
            if hermitian:
                tail, err = fit_hermitian_tail_on_window(
                    b, n_min=fit_min_n, n_max=fit_max_n, known_moments=known_moments,
                    n_tail_max=10 * len(b.mesh), expansion_order=fit_max_moment)
            else:
                tail, err = fit_tail_on_window(b, n_min=fit_min_n, n_max=fit_max_n, known_moments=known_moments,
                                               n_tail_max=10 * len(b.mesh), expansion_order=fit_max_moment)
            replace_by_tail(b, tail, n_min=fit_min_n)

    def _to_blockmatrix(self, number):
        bmat = dict()
        for bname, bsize in zip(self.blocknames, self.blocksizes):
            bmat[bname] = np.identity(bsize) * number
        return bmat

    def _quickplot(self, file_name, x_range=(0, 100)):
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
                plt.plot(mesh[ia:ie], b.data[ia:ie, i, j].real, ls='dashed')
        plt.savefig(file_name)
        plt.close()

    def flip_spin(self, blocklabel):
        up, dn = "up", "dn"
        self._checkforspins()
        if up in blocklabel:
            splittedlabel = blocklabel.split(up)
            new_label = splittedlabel[0] + dn + splittedlabel[1]
        elif dn in blocklabel:
            splittedlabel = blocklabel.split(dn)
            new_label = splittedlabel[0] + up + splittedlabel[1]
        assert isinstance(
            new_label, str), "couldn't flip spin, spins must be labeled up/dn"
        return new_label

    def _checkforspins(self):
        for name in self.blocknames:
            assert (len(name.split("up")) == 2) ^ (len(name.split("dn")) ==
                                                   2), "the strings up and dn must occur exactly once in blocknames"
