import unittest, os, numpy as np, itertools as itt

from bethe.models import NambuMomentumPlaquetteBethe


class TestModels(unittest.TestCase):

    def test_NambuMomentumPlaquetteBethe_initialization(self):
        model = NambuMomentumPlaquetteBethe(10, 1, 2, 2, 3)
        t_loc_ref = {"G": np.array([[7,0],[0,-7]]),
                     "X": np.array([[-3,0],[0,3]]),
                     "Y": np.array([[-3,0],[0,3]]),
                     "M": np.array([[-1,0],[0,1]])}
        for k, m in model.t_loc.items():
            for i, j in itt.product(range(2), range(2)):
                self.assertEqual(m[i, j], t_loc_ref[k][i, j])
