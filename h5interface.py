import os
from pytriqs.archive import HDFArchive
from pytriqs.operators import Operator
from pytriqs.applications.impurity_solvers.cthyb import AtomDiag
from pytriqs.gf.local import BlockGf
from pytriqs.utility import mpi


class Storage:
    """
    conveniently stores a bunch of objects into some default hdf5 structure
    takes care of the open/close status of the archive, especially in parallel computations
    internal functions have a leading underscore, avoid using them explicitly
    """
    def __init__(self, file_name, objects_to_store = {}):
        self.disk = None
        self.dmft_results = None
        self.file_name = file_name
        if mpi.is_master_node():
            self._open_archive()
            self._close_archive()
        self.memory_container = objects_to_store

    def _open_archive(self, read_only = False):
        assert mpi.is_master_node(), "Only the master node shall access the disk."
        assert not self._archive_is_open(), "Archive has already been opened."
        if read_only:
            self.disk = HDFArchive(self.file_name, 'r')
        else:
            self.disk = HDFArchive(self.file_name, 'a')
            if not self.disk.is_group("dmft_results"):
                self.disk.create_group("dmft_results")
                self.disk["dmft_results"]["n_dmft_loops"] = 0
        self.dmft_results = self.disk["dmft_results"]

    def _close_archive(self):
        del self.dmft_results
        self.dmft_results = None
        del self.disk
        self.disk = None

    def save_loop(self, objects_to_store = {}, *args):
        """args can be an arbitrary number of dicts"""
        for arg in args:
            objects_to_store.update(arg)
        self.memory_container.update(objects_to_store)
        new_loop_nr = self.get_completed_loops()
        if mpi.is_master_node():
            self._open_archive()
            self.dmft_results.create_group(str(new_loop_nr))
            place_to_store = self.dmft_results[str(new_loop_nr)]
            for name, obj in self.memory_container.items():
                if not obj is None:
                    place_to_store[name] = obj
            self.dmft_results["n_dmft_loops"] += 1
            self._close_archive()

    def _archive_is_open(self):
        if self.disk is None:
            return False
        else:
            return True

    def _asc_loop_nr(self, loop_nr):
        assert self._archive_is_open(), "Archive has to be open, when loop numbers are read"
        n_loops = self.dmft_results["n_dmft_loops"]
        if loop_nr is None:
            loop_nr = n_loops - 1
        if loop_nr < 0:
            loop_nr = n_loops + loop_nr
            assert loop_nr >= 0, "loop not available"
        return loop_nr

    def load(self, quantity_name, loop_nr = None, bcast = True):
        """
        allows for negative loop numbers counting backwards from the end
        don't bcast, if you want to load an AtomDiag of TRIQS
        """
        quantity = None
        if mpi.is_master_node():
            self._open_archive(True)
            loop_nr = self._asc_loop_nr(loop_nr)
            try:
                quantity = self.dmft_results[str(loop_nr)][quantity_name]
            except KeyError:
                pass
            self._close_archive()
        if bcast:
            quantity = mpi.bcast(quantity)
        return quantity

    def get_completed_loops(self, _archive_open = False):
        n_loops = None
        if mpi.is_master_node():
            if _archive_open:
                n_loops = self.dmft_results["n_dmft_loops"]
            else:
                self._open_archive(True)
                n_loops = self.dmft_results["n_dmft_loops"]
                self._close_archive()
        n_loops = mpi.bcast(n_loops)
        return n_loops

    def get_last_loop_nr(self):
        return self.get_completed_loops() - 1

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
        """
        deletes a loop
        """
        self._open_archive()
        loop = self._asc_loop_nr(loop)
        self._drop_loop(loop)
        for l in range(self.get_completed_loops(True)):
            if l > loop:
                self._relabel_loop(l, l - 1)
        self.disk["dmft_results"]["n_dmft_loops"] -= 1
        self._close_archive()

    def merge(self, storage_to_append):
        """
        appends a storage on another
        """
        n_loops_sto2 = storage_to_append.get_completed_loops()
        n_loops_sto = self.get_completed_loops()
        self._open_archive()
        storage_to_append._open_archive()
        for l in range(n_loops_sto2):
            appended_loop_nr = str(n_loops_sto + l)
            self.dmft_results.create_group(appended_loop_nr)
            for key, val in storage_to_append.dmft_results[str(l)].items():
                self.dmft_results[appended_loop_nr][key] = val
        self.dmft_results["n_dmft_loops"] +=  n_loops_sto2
        self._close_archive()
        storage_to_append._close_archive()        
            
    def provide_initial_guess(self, provide_mu = True):
        try:
            sigma = self.load("se_imp_iw")
        except KeyError:
            sigma = self.load("sigma_iw") #backward compatibility
        if provide_mu:
            mu = self.load("mu")
            return {"self_energy": sigma, "mu": mu}
        else:
            return {"self_energy": sigma}
