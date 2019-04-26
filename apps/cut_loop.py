import sys

from cdmft.h5interface import Storage

arch_name = sys.argv[1]
loop = int(sys.argv[2])
sto = Storage(arch_name)
sto.cut_loop(loop)
