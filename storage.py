import os
from pytriqs.archive import HDFArchive
from pytriqs.operators import Operator
from pytriqs.applications.impurity_solvers.cthyb import AtomDiag
from pytriqs.gf.local import BlockGf
from pytriqs.utility import mpi


class LoopStorage:

    def __init__(self, file_name, objects_to_store = {}):
        self.disk = None
        self.dmft_results = None
        self.file_name = file_name
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
                if not obj is None:
                    place_to_store[name] = obj
            self.dmft_results["n_dmft_loops"] += 1

    def _asc_loop_nr(self, loop_nr):
        if loop_nr is None:
            loop_nr = self.get_last_loop_nr()
        if loop_nr < 0:
            loop_nr = self.get_completed_loops() + loop_nr
            assert loop_nr >= 0, "loop not available"
        return loop_nr

    def load(self, quantity_name, loop_nr = None, bcast = True):
        quantity = None
        if mpi.is_master_node():
            loop_nr = self._asc_loop_nr(loop_nr)
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

    def provide(self, f_str):
        f = None
        if self.get_completed_loops() > 0:
            last_loop = self.get_last_loop_nr()
            if mpi.is_master_node():
                f = self.dmft_results[str(last_loop)][f_str]
        f = mpi.bcast(f)
        return f
        
    def _drop_loop(self, loop):
        results = self.disk["dmft_results"]
        del results[str(loop)]

    def _relabel_loop(self, old_label, new_label):
        old_label = str(old_label)
        new_label = str(new_label)
        results = self.disk["dmft_results"]
        assert not results.is_group(new_label), "unable to relabel, group already exists"
        results.create_group(new_label)
        for key, val in results[old_label].items():
            results[new_label][key] = val
        #self.disk["dmft_results"][new_label_str] = self.disk["dmft_results"][str(old_label)]
        del results[old_label]

    def cut_loop(self, loop):
        loop = self._asc_loop_nr(loop)
        self._drop_loop(loop)
        for l in range(self.get_completed_loops()):
            if l > loop:
                self._relabel_loop(l, l - 1)
        self.disk["dmft_results"]["n_dmft_loops"] -= 1

    def merge(self, storage_to_append):
        n_loops_sto2 = storage_to_append.get_completed_loops()
        n_loops_sto = self.get_completed_loops()
        for l in range(n_loops_sto2):
            appended_loop_nr = str(n_loops_sto + l)
            self.dmft_results.create_group(appended_loop_nr)
            for key, val in storage_to_append.dmft_results[str(l)].items():
                self.dmft_results[appended_loop_nr][key] = val
        self.dmft_results["n_dmft_loops"] +=  n_loops_sto2
            
