import sys

from bethe.storage import LoopStorage

arch_name = sys.argv[1]
loop = int(sys.argv[2])
sto = LoopStorage(arch_name)
sto.cut_loop(loop)
