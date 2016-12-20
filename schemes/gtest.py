from pytriqs.gf.local import GfImFreq, BlockGf


class GLocal(BlockGf):
    pass

class WeissField(BlockGf):
    pass

g1 = GLocal(name_list = ['a'], block_list = [GfImFreq(beta = 10, indices = range(1), n_points = 100)])
g2 = WeissField(name_list = ['a'], block_list = [GfImFreq(beta = 10, indices = range(1), n_points = 100)])
g1 += g2
