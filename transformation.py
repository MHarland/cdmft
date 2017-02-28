import numpy as np, itertools as itt
from pytriqs.gf.local import BlockGf, GfImFreq

from bethe.greensfunctions import MatsubaraGreensFunction


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
    where U is the transformation_matrix given in gf_struct block structure
    subsequent reblocking into gf_struct_new is optional
    TRIQS Blockstructure makes calculations more efficient. Values outside the blocks are zero. This
    transformation class supports a change of the blockstructure. It can be defined explicitly by
    reblock_map or be calculated automatically or be suppressed.
    orbital_filter allows to omit transformation on certain orbitals
    """
    def __init__(self, gf_struct, transformation_matrix = None, gf_struct_new = None, reblock_map = None, orbital_filter = []):
        self.gf_struct = gf_struct
        self.blocksizes = [len(block[1]) for block in self.gf_struct]
        self.mat = transformation_matrix
        self.gf_struct_new = gf_struct_new if gf_struct_new is not None else gf_struct
        self.gf_struct_names_new = [b[0] for b in self.gf_struct_new]
        self.reblock_map = reblock_map
        self.orbital_filter = orbital_filter

    def transform_matrix(self, matrix, reblock = True):
        result = {}
        for block in self.gf_struct:
            bname = block[0]
            if bname in self.orbital_filter:
                result[bname] = matrix[bname]
            else:
                result[bname] = self.mat[bname].dot(matrix[bname]).dot(self.mat[bname].transpose().conjugate())
        if reblock and self.reblock_map is not None:
            result = self.reblock_by_map(result, self.reblock_map)
        elif reblock:
            result = self.reblock(result, self.gf_struct, self.gf_struct_new)
        return result

    def backtransform_matrix(self, matrix, reblock = True):
        if reblock and self.reblock_map is not None:
            result = self.reblock_by_map(matrix, self.reblock_map, True)
        elif reblock:
            result = self.reblock(matrix, self.gf_struct_new, self.gf_struct)
        tmp = result.copy()
        result = {}
        for block in self.gf_struct:
            bname = block[0]
            if bname in self.orbital_filter:
                result[bname] = tmp[bname]
            else:
                result[bname] = self.mat[bname].transpose().conjugate().dot(tmp[bname]).dot(self.mat[bname])
        return result
    
    def transform_g(self, gf, reblock = True):
        blocknames = [ind for ind in gf.indices]
        #result = gf.__class__(self, gf_init = gf)
        result = MatsubaraGreensFunction(gf_init = gf)
        result.zero()
        for bname in blocknames:
            indices = [int(ind) for ind in gf[bname].indices]
            for i1, i2, j1, j2 in itt.product(*[indices]*4):
                if not(bname in self.orbital_filter):
                    result[bname][i1, i2] += self.mat[bname][i1, j1] * gf[bname][j1, j2] * self.mat[bname].transpose().conjugate()[j2, i2]
                elif i1 == j1 and i2 == j2:
                    result[bname][i1, i2] = gf[bname][j1, j2]
        if reblock and self.reblock_map is not None:
            result = self.reblock_by_map(result, self.reblock_map)
        elif reblock:
            result = self.reblock(result, self.gf_struct, self.gf_struct_new)
        return result

    def backtransform_g(self, gf, reblock = True):
        if reblock and self.reblock_map is not None:
            result = self.reblock_by_map(gf, self.reblock_map, backtransform = True)
        elif reblock:
            result = self.reblock(gf, self.gf_struct_new, self.gf_struct)
        tmp = result.copy()
        result.zero()
        blocknames = [ind for ind in result.indices]
        for bname in blocknames:
            indices = [int(ind) for ind in result[bname].indices]
            for i1, i2, j1, j2 in itt.product(*[indices]*4):
                if not(bname in self.orbital_filter):
                    result[bname][i1, i2] += self.mat[bname].transpose().conjugate()[i1, j1] * tmp[bname][j1, j2] * self.mat[bname][j2, i2]
                elif i1 == j1 and i2 == j2:
                    result[bname][i1, i2] = tmp[bname][j1, j2]
        return result

    def reblock(self, matrix, struct_old, struct_new):
        """values outside the new blockstructure are dropped, values missing in the (old) blockstructure are zero"""
        interface_matrix = InterfaceToBlockstructure(matrix, struct_old, struct_new)
        if isinstance(matrix, BlockGf):
            if type(matrix) == BlockGf:
                n_iw = int(len(matrix.mesh)*.5)
                result = BlockGf(name_block_generator = [(s, GfImFreq(beta = matrix.beta, n_points = n_iw, indices = b)) for s, b in struct_new])
            else:
                result = matrix.__class__(gf_struct = struct_new, beta = matrix.beta, n_iw = matrix.n_iw)
        else:
            result = dict([[block[0], np.zeros([len(block[1]), len(block[1])], dtype = matrix[matrix.keys()[0]].dtype)] for block in struct_new])
        for block in struct_new:
            b_new = block[0]
            indices_new = block[1]
            for i1_new, i2_new in itt.product(indices_new, indices_new):
                result[b_new][i1_new, i2_new] = interface_matrix[b_new, i1_new, i2_new]
        return result

    def reblock_by_map(self, matrix, map_dict, backtransform = False):
        """returns a new BlockGf with gf_struct_new. map_dict maps old 3-tupel (block, index1, index2) to a new 3-tupel"""
        if backtransform:
            map_dict = dict([(b, a) for a, b in self.reblock_map.items()])
        if isinstance(matrix, BlockGf):
            return self._reblock_gf_by_map(matrix, map_dict, backtransform)
        else:
            return self._reblock_matrix_by_map(matrix, map_dict, backtransform)

    def _reblock_gf_by_map(self, gf, map_dict, backtransform):
        if not backtransform:
            result = BlockGf(name_list = self.gf_struct_names_new, block_list = [GfImFreq(indices = block[1], mesh = gf.mesh) for block in self.gf_struct_new])
        else:
            result = BlockGf(name_list = [block[0] for block in self.gf_struct], block_list = [GfImFreq(indices = block[1], mesh = gf.mesh) for block in self.gf_struct])
        for old, new in map_dict.items():
            result[new[0]][new[1], new[2]] << gf[old[0]][old[1], old[2]]
        return result

    def _reblock_matrix_by_map(self, matrix, map_dict, backtransform):
        if not backtransform:
            result = dict([[block[0], np.zeros([len(block[1]), len(block[1])])] for block in self.gf_struct_new])
        else:
            result = dict([[block[0], np.zeros([len(block[1]), len(block[1])])] for block in self.gf_struct])
        for old, new in map_dict.items():
            result[new[0]][new[1], new[2]] = matrix[old[0]][old[1], old[2]]
        return result

class InterfaceToBlockstructure:
    """
    blocked_matrix m must be an object accessible via m[blockname][index1, index2]
    interface for reading data only, i.e. not writing data into the source
    if data outside blocks is being accessed, getitem returns 0
    order matters, that is why it is initialized by both structs
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
