import matplotlib as mpl, sys, numpy as np
mpl.use("PDF")
from matplotlib import pyplot as plt
from pytriqs.gf.local import GfReFreq, GfImTime, GfLegendre
from pytriqs.archive import HDFArchive
from pytriqs.statistics.histograms import Histogram


for archive_name in sys.argv[1:]:
    results_groupname = 'som_results'
    fig = plt.figure()
    ax = fig.add_axes([.12,.1+.5,.83,.82-.45])
    archive = HDFArchive(archive_name, 'r')
    histos = archive[results_groupname]['histograms']
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
    grec = archive[results_groupname]['g_rec_l']
    gori = archive[results_groupname]['g_l']
    mesh = [x.real for x in grec.mesh]
    if isinstance(gori, GfLegendre):
        ax.plot(mesh, np.log(abs(gori.data[:,0,0].real)), label = "original", marker = "+")
        ax.plot(mesh, np.log(abs(grec.data[:,0,0].real)), ls = '--', label = "reconstructed", marker = 'x')
        ax.set_xlabel("$l$")
        ax.set_ylabel("$\\mathrm{log}\,|G(l)|$")
    elif isinstance(gori, GfImTime):
        center = int(len(mesh)*.5)
        halfrange = int(len(mesh)*.1)
        pr = (center - halfrange, center + halfrange)
        ax.plot(mesh[pr[0]:pr[1]], gori.data[pr[0]:pr[1],0,0].real, label = "original", marker = "+")
        ax.plot(mesh[pr[0]:pr[1]], grec.data[pr[0]:pr[1],0,0].real, ls = '--', label = "reconstructed", marker = 'x')
        ax.set_xlabel("$\\tau$")
        ax.set_ylabel("$G(\\tau)$")
    ax.legend(fontsize = 8, loc = "upper right")
    
    outname = archive_name[:-3]+"_som_hist.pdf"
    plt.savefig(outname)
    print outname+" ready"
    plt.close()
