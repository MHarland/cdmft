import unittest, os, numpy as np
from bethe.greensfunctions import MatsubaraGreensFunction
from pytriqs.gf.local import BlockGf, GfImFreq, iOmega_n, inverse, GfImTime, delta, is_gf_real_in_tau
from pytriqs.archive import HDFArchive


class TestMatsubaraGreensFunction(unittest.TestCase):

    def test_MatsubaraGreensFunction_initialization(self):
        g = MatsubaraGreensFunction(['up', 'dn'], [2, 2], 10, 1000)
        g2 = MatsubaraGreensFunction(gf_init = g)
        g3 = BlockGf(name_list = ['up'], block_list = [GfImFreq(beta = 10, n_points = 1000, indices = range(2))])
        g2 = MatsubaraGreensFunction(gf_init = g3)

    def test_MatsubaraGreensFunction_hdf(self):
        g = MatsubaraGreensFunction(['up', 'dn'], [2, 2], 10, 1000)
        archive = HDFArchive('tmp.h5', 'w')
        archive['g'] = g.get_as_BlockGf()
        del archive
        os.remove('tmp.h5')

    def test_MatsubaraGreensFunction_make_g_tau_real(self):
        """
        TODO
        """
        g = MatsubaraGreensFunction(['up', 'dn'], [2, 2], 10, 1000)
        g_tau = BlockGf(name_list = ['up', 'dn'],
                        block_list = [GfImTime(beta = 10, indices = range(2),
                                               n_points = 10001) for s in [2, 2]])
        for s, b in g:
            b.data[:,:,:] = (np.random.rand(b.data.shape[0], b.data.shape[1], b.data.shape[2]) -.5)* .001
            b.data[:, :,:] += complex(0, 1) * (np.random.rand(b.data.shape[0], b.data.shape[1], b.data.shape[2]) -.5) * .001
        for s, b in g:
            self.assertTrue(not is_gf_real_in_tau(b))
        g.make_g_tau_real(10001)
        for bn, b in g:
            self.assertTrue(is_gf_real_in_tau(b))


    def test_MatsubaraGreensFunction_fit_tail2(self):
        g = MatsubaraGreensFunction(['up'], [1], 10, 100)
        g << 1.
        #g['up'].data[:,0,0] = np.array([1./n + np.random.randn() * 0.0001 for n in g.mesh])
        g.fit_tail2(5, 10, max_mom_to_fit = 3)

    def test_MatsubaraGreensFunction_basic_math(self):
        gtriqs = BlockGf(name_list = ['up'], block_list = [GfImFreq(beta = 10, n_points = 1000, indices = range(2))])
        g = MatsubaraGreensFunction(gf_init = gtriqs)
        g << 1
        g += gtriqs
        gtriqs += g
        g << gtriqs
        gtriqs << g
        g2 = MatsubaraGreensFunction(gf_init = gtriqs)
        g << 1
        self.assertEqual(g['up'].data[0,0,0], 1)
        g2 << 3
        g += g2
        self.assertEqual(g['up'].data[0,0,0], 4)
        g -= g2
        self.assertEqual(g['up'].data[0,0,0], 1)
        g << g2 + 2 * g2
        self.assertEqual(g['up'].data[0,0,0], 9)
        self.assertTrue(isinstance(g, MatsubaraGreensFunction))
