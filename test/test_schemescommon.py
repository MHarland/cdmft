import unittest

from cdmft.schemes.common import GLocalCommon, WeissFieldCommon, SelfEnergyCommon


class TestSchemesCommon(unittest.TestCase):

    def test_SchemesCommon_inits_and_basic_maths(self):
        g = GLocalCommon(['up', 'dn'], [2, 2], 10, 1001)
        se = SelfEnergyCommon(['up', 'dn'], [2, 2], 10, 1001)
        g0 = WeissFieldCommon(['up', 'dn'], [2, 2], 10, 1001)
        g << 1.
        se << 2.
        g0 << 3.
        g.calc_dyson(g0, se)
