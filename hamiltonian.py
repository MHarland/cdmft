from pytriqs.operators import c as C, c_dag as CDag, n as N

from gfoperations import sum


class HubbardSite:
    
    def __init__(self, u, spins):
        self.up = spins[0]
        self.dn = spins[1]
        self.h_int = u * N(self.up, 0) * N(self.dn, 0)

    def get_h_int(self):
        return self.h_int

    def get_gf_struct(self):
        return [[self.up, range(1)], [self.dn, range(1)]]


class HubbardPlaquette:

    def __init__(self, u, spins):
        self.up = spins[0]
        self.dn = spins[1]
        self.h_int = u * sum([N(self.up, i) * N(self.dn, i) for i in range(4)])

    def get_h_int(self):
        return self.h_int

    def get_gf_struct(self):
        return [[self.up, range(4)], [self.dn, range(4)]]
