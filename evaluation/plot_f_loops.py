import matplotlib, sys, numpy as np
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from Bethe.storage import LoopStorage


archive_name = sys.argv[1]
function = sys.argv[2]
block = sys.argv[3]
ind1 = int(sys.argv[4])
ind2 = int(sys.argv[5])
first_loop = int(sys.argv[6])
last_loop = int(sys.argv[7])
arg_min = int(sys.argv[8])
arg_max = int(sys.argv[9])
part = sys.argv[10]
assert part in ["real", "imag"], "only real or imag available"

archive = LoopStorage(archive_name)
fig = plt.figure()
ax = fig.add_axes([.12,.12,.85,.85])
for loop_nr in range(first_loop, last_loop + 1):
    f = archive.load(function, loop_nr)
    if loop_nr == first_loop:
        mesh = np.array([xi for xi in f.mesh]).imag
        arg_min += int(len(mesh) * .5)
        arg_max += int(len(mesh) * .5)
        x = np.array(mesh)[arg_min: arg_max]
    if part == "imag":
        y = f[block].data[arg_min: arg_max, ind1, ind2].imag
    elif part == "real":
        y = f[block].data[arg_min: arg_max, ind1, ind2].real
    ax.plot(x, y, label = loop_nr)
ax.set_xlim(mesh[arg_min], mesh[arg_max])
ax.legend(title = "loop")
plt.savefig(archive_name[:-3] + ".pdf")
plt.close()
