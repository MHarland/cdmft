import matplotlib, sys
matplotlib.use("PDF")
from matplotlib import pyplot as plt

from bethe.evaluation.nambumomentumplaquettebethe import Evaluation


for archive_name in sys.argv[1:]:
    ev = Evaluation(archive_name)
    data = ev.get_scorder_loop()
    fig = plt.figure()
    ax = fig.add_axes([.15,.15,.75,.75])
    ax.errorbar(data[:,0], data[:,1], yerr = data[:,2], marker = 'o', fillstyle = "none")
    ax.set_xlabel("$\mathrm{DMFT-loop}$")
    ax.set_ylabel("$\Psi_{dSC}$")
    plt.savefig(archive_name[:-3]+"_scoloop.pdf")
    plt.close()
