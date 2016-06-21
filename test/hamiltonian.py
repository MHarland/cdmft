import unittest

from Bethe.hamiltonian import HubbardSite


class TestHamiltonians(unittest.TestCase):

    def test_HubbardSite(self):
        h = HubbardSite(3, ["up", "dn"])
        h_int = h.get_h_int()
