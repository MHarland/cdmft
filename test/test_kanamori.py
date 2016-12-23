import unittest, numpy as np


from bethe.operators.kanamori import Dimer


class TestKanamori(unittest.TestCase):

    def test_KanamoriDimer(self):
        ham = Dimer(1, 2)
        h_int = ham.get_h_int()
        ham.add_field_sz(.5)
        ham.add_field_sz(.5)
        self.assertTrue(ham.field_sz)
        ham.rm_field_sz(.5)
        self.assertTrue(not ham.field_sz)
