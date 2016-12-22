import itertools as itt
from pytriqs.operators.util.hamiltonians import h_int_kanamori
from pytriqs.operators.util.U_matrix import U_matrix_kanamori


class Dimer:
    def __init__(self, u, j, spins = ['up', 'dn'], orbs = ['d', 'c']):
        self.u = u
        self.j = j
        self.spins = spins
        self.orbs = orbs

    def get_h_int(self):
        umat, upmat = U_matrix_kanamori(2, self.u, self.j)
        mos = {(sn, on): (str(sn)+"-"+str(on), 0) for sn, on in itt.product(self.spins, self.orbs)}
        h = h_int_kanamori(self.spins, self.orbs, umat, upmat, self.j, map_operator_structure = mos)
        mos = {(sn, on): (str(sn)+"-"+str(on), 1) for sn, on in itt.product(self.spins, self.orbs)}
        h += h_int_kanamori(self.spins, self.orbs, umat, upmat, self.j, map_operator_structure = mos)
        return h

    def get_gf_struct(self):
        return [(s+'-'+orb, range(2)) for s, orb in itt.product(self.spins, self.orbs)]
