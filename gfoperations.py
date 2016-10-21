import itertools as itt, numpy as np
from pytriqs.gf.local import BlockGf, GfLegendre


def double_dot_product(matrix1, gf, matrix2):
    inds = range(matrix1.shape[0])
    prod = gf.copy()
    for i, l in itt.product(inds, inds):
        prod[i, l] = sum([matrix1[i, j] * gf[j, k] * matrix2[k, l] for j, k in itt.product(inds, inds)])
    return prod
            
def dot_product(matrix, gf):
    inds = range(matrix.shape[0])
    prod = gf.copy()
    for i, k in itt.product(inds, inds):
        prod[i, k] = sum([matrix[i, j] * gf[j, k] for j in inds])
    return prod
            
def sum(summands_list):
    result = 0
    for el in summands_list:
        result += el
    return result

def trace(block_gf, tr_gf):
    tr_gf.zero()
    n = 0
    for s, b in block_gf:
        for i in b.indices:
            n += 1
            tr_gf << tr_gf + b[i, i]
    tr_gf << tr_gf /float(n)

def cut_coefficients(glegendre, n_remaining_coeffs):
    g_cut = GfLegendre(indices = [i for i in glegendre.indices], beta = glegendre.beta, n_points = n_remaining_coeffs)
    g_cut.data[:,:,:] = glegendre.data[:n_remaining_coeffs,:,:]
    g_cut.enforce_discontinuity(np.identity(g_cut.N1))
    return g_cut
