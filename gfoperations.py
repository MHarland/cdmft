import itertools as itt


def double_dot_product(matrix1, gf, matrix2):
    inds = range(matrix1.shape[0])
    prod = gf.copy()
    for i, l in itt.product(inds, inds):
        prod[i, l] = sum([matrix1[i, j] * gf[j, k] * matrix2[k, l] for j, k in itt.product(inds, inds)])
    return prod
            
def sum(summands_list):
    result = 0
    for el in summands_list:
        result += el
    return result
