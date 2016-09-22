import sys

from bethe.storage import LoopStorage

sto_target = sys.argv[1]
sto_to_append = sys.argv[2]
sto0 = LoopStorage(sto_target)
sto1 = LoopStorage(sto_to_append)
sto0.merge(sto1)

