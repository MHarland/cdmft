import sys

from bethe.h5interface import Storage

arch_name = sys.argv[1]
first_loop = int(sys.argv[2])
last_loop = int(sys.argv[3])
sto = Storage(arch_name)
for loop in range(last_loop, first_loop -1, -1):
    sto.cut_loop(loop)
