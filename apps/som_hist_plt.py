import matplotlib as mpl, sys, numpy as np
mpl.use("PDF")
from matplotlib import pyplot as plt
from pytriqs.gf.local import GfReFreq
from pytriqs.archive import HDFArchive
from pytriqs.statistics.histograms import Histogram


for archive_name in sys.argv[1:]:
    fig = plt.figure()
    ax = fig.add_axes([.12,.1+.5,.83,.82-.45])
    archive = HDFArchive(archive_name, 'r')
    histos = archive['som_results']['histograms']
    hist = histos[0]
    dx = (hist.limits[1] - hist.limits[0]) / len(hist)
    x = np.linspace(hist.limits[0], hist.limits[1], len(hist.data))
    y = hist.data
    ax.bar(x, y, dx, color = 'gray', linewidth = 0.1)
    ax.set_xlabel("$D$")
    ax.set_ylabel("$P(D)$")
    ax.set_xlim(*hist.limits)
    ax.set_ylim(bottom=0)

    ax = fig.add_axes([.12,.1,.83,.82-.45])
    grec = archive['som_results']['g_rec_l']
    gori = archive['som_results']['g_l']
    mesh = [x.real for x in grec.mesh]
    ax.plot(mesh, np.log(abs(gori.data[:,0,0].real)), label = "original", marker = "+")
    ax.plot(mesh, np.log(abs(grec.data[:,0,0].real)), ls = '--', label = "reconstructed", marker = 'x')
    ax.set_xlabel("$l$")
    ax.set_ylabel("$\\mathrm{log}\,|G(l)|$")
    ax.legend(fontsize = 8, loc = "upper right")
    
    outname = archive_name[:-3]+"_som_hist.pdf"
    plt.savefig(outname)
    print outname+" ready"
    plt.close()
