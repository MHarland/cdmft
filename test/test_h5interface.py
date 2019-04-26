import unittest, os
from pytriqs.utility import mpi

from cdmft.h5interface import Storage


class TestStorage(unittest.TestCase):
    
    def test_Storage_initialization(self):
        sto = Storage("test.h5")
        if mpi.is_master_node(): os.remove("test.h5")

    def test_Storage_get_completed_loops(self):
        sto = Storage("test.h5")
        self.assertEqual(sto.get_completed_loops(), 0)
        if mpi.is_master_node: os.remove("test.h5")

    def test_Storage_save_load_cut_merge(self):
        sto = Storage("test.h5")
        store_me = [1,2,3]
        sto.save_loop({'store_me': store_me, 'store_me3': 2.})
        sto.save_loop()
        load_me = sto.load('store_me')
        load_me3 = sto.load('store_me3')
        self.assertEqual(sto.get_completed_loops(), 2)
        self.assertEqual(len(load_me), 3)
        self.assertEqual(load_me3, 2.)
        load_me_not = sto.load('store_me2')
        self.assertTrue(load_me_not is None)
        sto.cut_loop(-1)
        self.assertEqual(sto.get_completed_loops(), 1)
        sto2 = Storage("test2.h5")
        sto2.save_loop({'store_me': store_me})
        sto.merge(sto2)
        self.assertEqual(sto.get_completed_loops(), 2)
        if mpi.is_master_node(): os.remove("test.h5")
        if mpi.is_master_node(): os.remove("test2.h5")
