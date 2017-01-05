import unittest, numpy as np
from pytriqs.operators import Operator

from bethe.operators.kanamori import Dimer


class TestKanamori(unittest.TestCase):

    def test_KanamoriDimer(self):
        ham = Dimer(5, 3)
        h_int = ham.get_h_int()
        ham.set_h_int(5, 3)
        h_int2 = ham.get_h_int()
        self.assertEqual(str(h_int - h_int2), '0')
        ham.add_field_sz(.5)
        ham.add_field_sz(.5)
        self.assertTrue(ham.gap_sz is not None)
        ham.rm_field_sz()
        self.assertTrue(ham.gap_sz is None)
