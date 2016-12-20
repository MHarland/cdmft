import unittest, numpy as np

from bethe.operators.hubbard import Site, Plaquette, PlaquetteMomentum, PlaquetteMomentumNambu, TriangleMomentum


class TestHubbard(unittest.TestCase):

    def test_HubbardSite(self):
        h = Site(3, ["up", "dn"])
        h_int = h.get_h_int()

    def test_HubbardPlaquette(self):
        h = Plaquette(3, ["up", "dn"])
        h_int = h.get_h_int()

    def test_HubbardPlaquetteMomentum(self):
        h = PlaquetteMomentum(3)
        h_int = h.get_h_int()
        self.assertEqual(h._to_mom.index_map, {('dn', 1): ('dn-X', 0), ('up', 1): ('up-X', 0), ('dn', 0): ('dn-G', 0), ('up', 2): ('up-Y', 0), ('up', 3): ('up-M', 0), ('dn', 3): ('dn-M', 0), ('dn', 2): ('dn-Y', 0), ('up', 0): ('up-G', 0)})

    def test_HubbardPlaquetteMomentumNambu(self):
        transf = {"up": .5 * np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]),
                  "dn": .5 * np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]])}
        h = PlaquetteMomentumNambu(4, ["up", "dn"], ["G", "X", "Y", "M"], transf)

    def test_HubbardTriangleMomentum(self):
        h = TriangleMomentum(3)
        h_int = h.get_h_int()
