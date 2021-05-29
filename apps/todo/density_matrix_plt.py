import numpy as np, sys

from cdmft.evaluation.common import Evaluation
from cdmft.h5interface import Storage
from cdmft.plot.cfg import plt


for arch in sys.argv[1:]:
    sto = Storage(arch)
    ev = Evaluation(sto)
    roh = ev.get_density_matrix()
    fig = plt.figure()
    ax = fig.add_axes([.1,.1,.8,.75])
    ax.matshow(roh)
    ax.set_title("$\\rho$")
    plt.savefig(arch[:-3]+"_density_matrix.pdf")
    plt.close()
