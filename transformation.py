import numpy as np, itertools as itt
from pytriqs.gf.local import BlockGf, GfImFreq


class GfStructTransformationIndex:
    """Supports transformation of vectors. They don't depend on a 2D blockstructure,
    but indices are grouped differently from struct to struct. Used to transform 
    operators of hamiltonians.
    """
    def __init__(self, gf_struct_new, gf_struct_old):
        self.gf_struct_new = gf_struct_new
        self.new_blocksizes = [len(b[1]) for b in self.gf_struct_new]
        self.gf_struct_old = gf_struct_old
        self.old_blocksizes = [len(b[1]) for b in self.gf_struct_old]
        assert np.sum(self.new_blocksizes) == np.sum(self.old_blocksizes), "struct sizes do not match"
        self.index_map = {}
        index_new = 0
        blocknr_new = 0
        for blocknr, block in enumerate(self.gf_struct_old):
            old_block_name = block[0]
            for index_old in block[1]:
                new_block_name = self.gf_struct_new[blocknr_new][0]
                self.index_map.update({(old_block_name, index_old): (new_block_name, index_new)})
                index_new += 1
                if index_new == self.new_blocksizes[blocknr_new]:
                    blocknr_new += 1
                    index_new = 0

    def __call__(self, block, index):
        return self.index_map[(block, index)]


class MatrixTransformation:
    """
    G mapsto UGU^dag
    if backtransform: G mapsto U^dagGU
    where U is the transformation_matrix given in gf_struct block structure
    subsequent reblocking into gf_struct_new is optional
    """
    def __init__(self, gf_struct, transformation_matrix, gf_struct_new = None):
        self.gf_struct = gf_struct
        self.blocksizes = [len(block[1]) for block in self.gf_struct]
        self.mat = transformation_matrix
        self.gf_struct_new = gf_struct_new
        self.gf_struct_names_new = [b[0] for b in self.gf_struct_new]

    def transform_matrix(self, matrix):
        result = {}
        for block in self.gf_struct:
            bname = block[0]
            result[bname] = self.mat[bname].dot(matrix[bname]).dot(self.mat[bname].transpose().conjugate())
        return result

    def transform_gf_iw(self, gf, backtransform = False):
        blocknames = [ind for ind in gf.indices]
        result = BlockGf(name_list = self.gf_struct_names_new, block_list = [GfImFreq(indices = block[1], mesh = gf.mesh) for block in self.gf_struct_new])
        result.zero()
        for bname in blocknames:
            indices = [int(ind) for ind in gf[bname].indices]
            for i1, i2, j1, j2 in itt.product(*[indices]*4):
                if backtransform:
                    result[bname][i1, i2] += self.mat[bname].transpose().conjugate()[i1, j1] * gf[bname][j1, j2] * self.mat[bname][j2,i2]
                else:
                    result[bname][i1, i2] += self.mat[bname][i1, j1] * gf[bname][j1, j2] * self.mat[bname].transpose().conjugate()[j2,i2]
        return result

    def reblock(self, matrix):
        """values outside the new blockstructure are dropped, values missing in the (old) blockstructure are zero"""
        interface_matrix = InterfaceToBlockstructure(matrix, self.gf_struct, self.gf_struct_new)
        if isinstance(matrix, BlockGf):
            result = BlockGf(name_list = self.gf_struct_names_new, block_list = [GfImFreq(indices = block[1], mesh = matrix.mesh) for block in self.gf_struct_new])
        else:
            result = dict([[block[0], np.zeros([len(block[1]), len(block[1])])] for block in self.gf_struct_new])
        for block in self.gf_struct_new:
            b_new = block[0]
            indices_new = block[1]
            for i1_new, i2_new in itt.product(indices_new, indices_new):
                result[b_new][i1_new, i2_new] = interface_matrix[b_new, i1_new, i2_new]
        return result

    def reblock_by_map(self, matrix, map_dict):
        """returns a new BlockGf with gf_struct_new. map_dict maps old 3-tupel (block, index1, index2) to a new 3-tupel"""
        if isinstance(matrix, BlockGf):
            return self._reblock_gf_by_map(matrix, map_dict)
        else:
            return self._reblock_matrix_by_map(matrix, map_dict)

    def _reblock_gf_by_map(self, gf, map_dict):
        result = BlockGf(name_list = self.gf_struct_names_new, block_list = [GfImFreq(indices = block[1], mesh = matrix.mesh) for block in self.gf_struct_new])
        for old, new in map_dict.items():
            result[new[0]][new[1], new[2]] << matrix[old[0]][old[1], old[2]]
        return result

    def _reblock_matrix_by_map(self, matrix, map_dict):
        result = dict([[block[0], np.zeros([len(block[1]), len(block[1])])] for block in self.gf_struct_new])
        for old, new in map_dict.items():
            result[new[0]][new[1], new[2]] = matrix[old[0]][old[1], old[2]]
        return result

class InterfaceToBlockstructure:
    """
    blocked_matrix m must be an object accessible via m[blockname][index1, index2]
    interface for reading data only, i.e. not writing data into the source
    if data outside blocks is being accessed, getitem returns 0
    """
    def __init__(self, blocked_matrix, struct_old, struct_new):
        self.source = blocked_matrix
        self.struct_source = struct_old
        self.absolute_size = np.sum(self._blocksizes(self.struct_source), axis = 0)
        self.struct = struct_new
        self.struct_names = [b[0] for b in self.struct]

    def __getitem__(self, (block, i1, i2)):
        """gets coordinates in struct_new basis and returns corresponding value of source"""
        assert block in self.struct_names, "block "+block+" not in "+str(self.struct_names)
        j1, j2 = self._unblocked_position(block, i1, i2, self.struct)
        assert j1 < self.absolute_size and j2 < self.absolute_size, "indices outside of source space"
        src_pos = self._blocked_position(j1, j2, self.struct_source)
        if src_pos is None:
            return 0
        block, i1, i2 = src_pos
        return self.source[block][i1, i2]

    def _unblocked_position(self, block, i1, i2, struct):
        """gets block coords and returns absolute coords"""
        assert block in self.struct_names, block+" is not in "+str(self.struct_names)
        bsizes = self._blocksizes(struct)
        j = 0
        ind_block = 0
        block_found = False
        while not block_found:
            if block == struct[ind_block][0]:
                block_found = True
            else:
                j += bsizes[ind_block]
                ind_block += 1
        assert i1 < bsizes[ind_block] and i2 < bsizes[ind_block], block+", "+str(i1)+", "+str(i2)+" lies outside of any block"
        return j + i1, j + i2

    def _blocked_position(self, j1, j2, struct):
        """gets absolute coords and returns block coords, returns None if absolute position j1, j2 lies outside of the blockstructure"""
        bsizes = self._blocksizes(struct)
        j = j1 if j1 <= j2 else j2
        ind_block = 0
        pos_block = 0
        block_found = False
        while not block_found:
            if j < bsizes[ind_block]:
                block_found = True
            else:
                j -= bsizes[ind_block]
                pos_block += bsizes[ind_block]
                ind_block += 1
        block = struct[ind_block][0] 
        i1 = j1 - pos_block
        i2 = j2 - pos_block
        if i1 >= bsizes[ind_block] or i2 >= bsizes[ind_block]:
            return None
        return block, i1, i2

    def _blocksizes(self, struct):
        return [len(b[1]) for b in struct]
