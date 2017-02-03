import matplotlib as mpl, sys, numpy as np
#from pytriqs.gf.local import GfReFreq, GfImTime, GfLegendre
#from pytriqs.archive import HDFArchive
#from pytriqs.statistics.histograms import Histogram

from bethe.h5interface import Storage
from bethe.plot.cfg import plt


for archive_name in sys.argv[1:]:
    fig = plt.figure()
    ax = fig.add_axes([.12,.12,.83,.82])
    archive = Storage(archive_name)
    histos_orb = archive.load('perturbation_order')
    histo_tot = archive.load('perturbation_order_total')
    hist = histo_tot
    i_max = len(hist)
    for i in range(1, len(hist)):
        pt = hist[-i]
        if np.allclose(pt, 0):
            i_max -= 1
        else:
            break
    i_min = 0
    for i in range(len(hist)):
        pt = hist[i]
        if np.allclose(pt, 0):
            i_min += 1
        else:
            break
    x = np.linspace(i_min, i_max, i_max - i_min, True)
    n_bins = i_max - i_min
    n_orb = len(histos_orb)
    colors = [mpl.cm.jet(i/float(max(1, n_orb -1))) for i in range(n_orb)]
    xs = []
    ys = []
    labels =[]
    for (orbital, hist), color in zip(histos_orb.items(), colors):
        y = hist[i_min:i_max]
        xs.append(x)
        ys.append(y)
        labels.append('$\mathrm{'+orbital+'}$')
    ax.hist(xs, n_bins, weights = ys, histtype = 'step', color = colors, label = labels,
            stacked = True, normed = True, fill = False)
    ax.set_ylim(bottom = 0)
    ax.set_xlabel("$D$")
    ax.set_ylabel("$P(D)$")
    ax.legend()
    outname = archive_name[:-3]+"_pert_hist.pdf"
    plt.savefig(outname)
    print outname+" ready"
    plt.close()
