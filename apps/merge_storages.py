import sys

from bethe.h5interface import Storage

sto_target = sys.argv[1]
sto_to_append = sys.argv[2]
sto0 = Storage(sto_target)
sto1 = Storage(sto_to_append)
sto0.merge(sto1)

