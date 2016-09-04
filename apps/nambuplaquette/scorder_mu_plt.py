import matplotlib, sys
matplotlib.use("PDF")
from matplotlib import pyplot as plt
from pytriqs.gf.local import BlockGf

from bethe.evaluation.nambumomentumplaquettebethe import Evaluation
from bethe.storage import LoopStorage


dens = []
aves = []
stds = []
for archive_name in sys.argv[1:]:
    print "loading "+archive_name+"..."
    ev = Evaluation(archive_name)
    ave, std = ev.get_scorder()
    aves.append(ave)
    stds.append(std)
    try:
        d = ev.get_density()
        sto = LoopStorage(archive_name)
        d = sto.provide("mu")
        if isinstance(d, dict):
            for v in d.values():
                d = v[0,0]
    except KeyError:
        sto = LoopStorage(archive_name)
        g = sto.provide("g_loc_iw")
        d = g.total_density()
    dens.append(d)

fig = plt.figure()
ax = fig.add_axes([.15,.15,.8,.8])
ax.errorbar(dens, aves, yerr = stds, marker = 'o', fillstyle = "none")
ax.set_xlabel("$\\mu$")
ax.set_ylabel("$\Psi_{dSC}$")
ax.set_ylim(bottom = 0)
ax.set_xlim(min(dens), max(dens))
plt.savefig("sco_dens.pdf")
print "sco_dens.pdf ready"
