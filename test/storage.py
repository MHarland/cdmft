import unittest, os

from Bethe.storage import LoopStorage


class TestLoopStorage(unittest.TestCase):
    
    def test_loopstorage_initialization(self):
        sto = LoopStorage("test.h5")
        os.remove("test.h5")

    def test_loopstorage_get_completed_loops(self):
        sto = LoopStorage("test.h5")
        self.assertEqual(sto.get_completed_loops(), 0)
        os.remove("test.h5")
