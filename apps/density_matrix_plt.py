import matplotlib, numpy as np, sys
matplotlib.use("PDF")
from matplotlib import pyplot as plt

from bethe.evaluation.generic import Evaluation


for arch in sys.argv[1:]:
    ev = Evaluation(arch)
    roh = ev.get_density_matrix()
    fig = plt.figure()
    ax = fig.add_axes([.1,.1,.85,.85])
    ax.matshow(roh)
    ax.set_title("$roh$")
    plt.savefig(arch[:-3]+"_density_matrix.pdf")
    plt.close()
