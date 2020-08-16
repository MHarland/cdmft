import numpy as np
import itertools as itt
from pytriqs.gf import BlockGf, GfImFreq

from cdmft.greensfunctions import MatsubaraGreensFunction


class Transformation:
    """operations is a list of Reblock and UnitaryMatrixTransformation in the order that is to be performed"""

    def __init__(self, operations):
        self.ops = operations

    def transform(self, x):
        for op in self.ops:
            x = op(x)
        return x

    def backtransform(self, x):
        for op in self.ops[::-1]:
            x = op.inverse(x)
        return x


class Reblock:
    """reblock_map maps old to new"""

    def __init__(self, struct_new, struct_old, reblock_map):
        self.new = struct_new
        self.old = struct_old
        self.rmap = reblock_map

    def __call__(self, x):
        if isinstance(x, BlockGf):
            y = self._transform_gf(x)
        else:
            y = self._transform(x)
        return y

    def inverse(self, x):
        if isinstance(x, BlockGf):
            y = self._inverse_gf(x)
        else:
            y = self._inverse(x)
        return y

    def _transform(self, x):
        result = {b[0]: np.zeros(
            [len(b[1]), len(b[1])], dtype=complex) for b in self.new}
        for bn, b in x.items():
            for i, j in itt.product(range(b.shape[0]), range(b.shape[1])):
                if not ((bn, i, j) in self.rmap.keys()):
                    assert np.allclose(
                        x[bn][i, j], 0), "reblocking omits entry "+bn+str(i)+str(j)
        for old, new in self.rmap.items():
            result[new[0]][new[1], new[2]] = x[old[0]][old[1], old[2]]
        return result

    def _inverse(self, x):
        result = {b[0]: np.zeros([b[1], b[1]], dtype=complex)
                  for b in self.old}
        for bn, b in x.items():
            for i, j in itt.product(range(b.shape[0]), range(b.shape[1])):
                if not ((bn, i, j) in self.rmap.values()):
                    assert np.allclose(
                        x[bn][i, j], 0), "reblocking omits entry "+bn+str(i)+str(j)
        for old, new in self.rmap.items():
            result[old[0]][old[1], old[2]] = x[new[0]][new[1], new[2]]
        return result

    def _transform_gf(self, g):
        result = BlockGf(name_list=[b[0] for b in self.new], block_list=[
                         GfImFreq(indices=b[1], mesh=g.mesh) for b in self.new])
        all_indices = list()
        for bn, b in g:
            blockinds = [i for i in range(b.data.shape[1])]
            for i, j in itt.product(*[blockinds] * 2):
                all_indices.append((bn, i, j))
        for b, i, j in all_indices:
            if not((b, int(i), int(j)) in self.rmap.keys()):
                data = g[b].data[:, i, j]
                assert np.allclose(
                    np.sum(data), 0), "reblocking omits entry "+b+str(i)+str(j)
        for old, new in self.rmap.items():
            result[new[0]][new[1], new[2]] << g[old[0]][old[1], old[2]]
        return result

    def _inverse_gf(self, g):
        result = BlockGf(name_list=[b[0] for b in self.old], block_list=[
                         GfImFreq(indices=b[1], mesh=g.mesh) for b in self.old])
        all_indices = list()
        for bn, b in g:
            blockinds = [i for i in range(b.data.shape[1])]
            for i, j in itt.product(*[blockinds] * 2):
                all_indices.append((bn, i, j))
        for b, i, j in all_indices:
            if not((b, int(i), int(j)) in self.rmap.values()):
                data = g[b].data[:, i, j]
                assert np.allclose(
                    np.sum(data), 0), "reblocking omits entry "+b+str(i)+str(j)
        for old, new in self.rmap.items():
            result[old[0]][old[1], old[2]] << g[new[0]][new[1], new[2]]
        return result


class UnitaryMatrixTransformation:
    """ maps g to U g Udagger"""

    def __init__(self, matrix):
        self.mat = matrix

    def __call__(self, x):
        if isinstance(x, BlockGf):
            y = self._transform_gf(x)
        else:
            y = self._transform(x)
        return y

    def inverse(self, x):
        if isinstance(x, BlockGf):
            y = self._inverse_gf(x)
        else:
            y = self._inverse(x)
        return y

    def _transform(self, x):
        prod = {}
        for bn, b in x.items():
            prod[bn] = self.mat[bn].dot(b).dot(
                np.transpose(self.mat[bn]).conjugate())
        return prod

    def _inverse(self, x):
        prod = {}
        for bn, b in x.items():
            prod[bn] = np.transpose(
                self.mat[bn]).conjugate().dot(b).dot(self.mat[bn])
        return prod

    def _transform_gf(self, x):
        prod = x.copy()
        prod.zero()
        for bn, b in x:
            norbs = b.data.shape[1]
            orbs = range(norbs)
            for i, j, k, l in itt.product(*[orbs]*4):
                prod[bn][i, l] += self.mat[bn][i, j] * x[bn][j, k] * \
                    np.transpose(self.mat[bn])[k, l].conjugate()
        return prod

    def _inverse_gf(self, x):
        prod = x.copy()
        prod.zero()
        for bn, b in x:
            norbs = b.data.shape[1]
            orbs = range(norbs)
            for i, j, k, l in itt.product(*[orbs]*4):
                prod[bn][i, l] += np.transpose(self.mat[bn])[i,
                                                             j].conjugate() * x[bn][j, k] * self.mat[bn][k, l]
        return prod
