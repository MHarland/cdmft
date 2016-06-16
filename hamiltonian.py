from pytriqs.operators import c as C, c_dag as CDag, n as N

class HubbardSite:
    def __init__(self, u, block_names):
        up = block_names[0]
        dn = block_names[1]
        self.h_int = u * N(up, 0) * N(dn, 0)

    def get_h_int(self):
        return self.h_int
