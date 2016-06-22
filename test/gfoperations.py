import unittest, numpy as np, itertools as itt

from bethe.gfoperations import double_dot_product, sum

class TestGfOperations(unittest.TestCase):

    def test_sum(self):
        summands = [1,2,3]
        self.assertEqual(sum(summands), 6)

    def test_double_dot_product_2by2(self):
        a = np.array([[1,2],[0,3]])
        b = np.array([[4,5],[6,7]])
        c = np.array([[8,9],[10,11]])
        x1 = double_dot_product(a, b, c)
        x2 = a.dot(b.dot(c))
        inds = range(x1.shape[0])
        for i, j in itt.product(inds, inds):
            self.assertEqual(x1[i, j], x2[i, j])
