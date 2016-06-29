import os
from pytriqs.archive import HDFArchive
from pytriqs.gf.local import BlockGf
from pytriqs.utility import mpi


class LoopStorage:

    def __init__(self, file_name, objects_to_store = {}):
        self.disk = None
        self.dmft_results = None
        if os.path.isfile(file_name):
            new_archive = False
        else:
            new_archive = True
        if mpi.is_master_node():
            self.disk = HDFArchive(file_name, 'a')
            if new_archive:
                self.disk.create_group("dmft_results")
                self.dmft_results = self.disk["dmft_results"]
                self.dmft_results["n_dmft_loops"] = 0
            else:
                self.dmft_results = self.disk["dmft_results"]
        self.memory_container = objects_to_store

    def save_loop(self, objects_to_store = {}, *args):
        """args can be an arbitrary number of dicts"""
        for arg in args:
            objects_to_store.update(arg)
        self.memory_container.update(objects_to_store)
        new_loop_nr = self.get_completed_loops()
        if mpi.is_master_node():
            self.dmft_results.create_group(str(new_loop_nr))
            place_to_store = self.dmft_results[str(new_loop_nr)]
            for name, obj in self.memory_container.items():
                place_to_store[name] = obj
            self.dmft_results["n_dmft_loops"] += 1

    def load(self, quantity_name, loop_nr = None, bcast = True):
        quantity = None
        if loop_nr is None:
            loop_nr = self.get_last_loop_nr()
        if mpi.is_master_node():
            quantity = self.dmft_results[str(loop_nr)][quantity_name]
        if bcast:
            quantity = mpi.bcast(quantity)
        return quantity

    def get_completed_loops(self):
        n_loops = None
        if mpi.is_master_node():
            n_loops = self.dmft_results["n_dmft_loops"]
        n_loops = mpi.bcast(n_loops)
        return n_loops

    def get_last_loop_nr(self):
        return self.get_completed_loops() - 1

    def provide_last_g_loc(self):
        g_loc = None
        if self.get_completed_loops() > 0:
            last_loop = self.get_last_loop_nr()
            if mpi.is_master_node():
                g_loc = self.dmft_results[str(last_loop)]["g_loc_iw"]
        g_loc = mpi.bcast(g_loc)
        return g_loc
