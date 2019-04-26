import unittest

from cdmft.schemes.generic import GLocalGeneric, WeissFieldGeneric, SelfEnergyGeneric


class TestSchemesGeneric(unittest.TestCase):

    def test_SchemesGeneric_inits_and_basic_maths(self):
        g = GLocalGeneric(['up', 'dn'], [2, 2], 10, 1001)
        se = SelfEnergyGeneric(['up', 'dn'], [2, 2], 10, 1001)
        g0 = WeissFieldGeneric(['up', 'dn'], [2, 2], 10, 1001)
        g << 1.
        se << 2.
        g0 << 3.
        g.calc_dyson(g0, se)
