import unittest, numpy as np
from pytriqs.operators import Operator

from bethe.operators.kanamori import Dimer


class TestKanamori(unittest.TestCase):

    def test_KanamoriDimer(self):
        ham = Dimer(5, 3)
        h_int = ham.get_h_int()
        ham.add_field_sz(.51)
        ham.add_field_sz(.52)
        h_int = ham.get_h_int()
        self.assertTrue(ham.gap_sz is not None)
        ham.rm_field_sz()
        self.assertTrue(ham.gap_sz is None)
        op = ham.sz_tot()
        op = ham.s2_tot()
