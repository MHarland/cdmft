import matplotlib, sys, numpy as np

from bethe.h5interface import Storage
from bethe.plot.cfg import plt, ax
from bethe.setups.bethelattice import TriangleBetheSetup as Setup

n_colors = len(sys.argv[1:])
colors = [matplotlib.cm.jet(i/float(max(1,n_colors-1))) for i in range(n_colors)]
for fname, c in zip(sys.argv[1:], colors):
    w_max = 10
    sto = Storage(fname)
    g_mom = sto.load("se_imp_iw")
    setup = Setup(g_mom.beta, 0, 0, 0, 0)
    g = setup.transf.backtransform_g(g_mom)
    supermesh = np.array([iw.imag for iw in g.mesh])
    n_iw0 = int(len(supermesh)*.5)
    n_w_max = np.argwhere(supermesh <= w_max)[-1,0]
    mesh = supermesh[n_iw0:n_w_max]
    y = g["up"][0, 0].data[n_iw0:n_w_max,0,0].imag
    ax.plot(mesh, y, label = '$\mathrm{'+fname[:-3]+', loc}$', color = c, ls = 'dotted')
    y = g["up"][0, 1].data[n_iw0:n_w_max,0,0].imag
    ax.plot(mesh, y, label = '$\mathrm{'+fname[:-3]+', nn}$', color = c)
ax.set_xlabel("$i\\omega_n$")
ax.set_ylabel("$\\Sigma(i\\omega_n)$")
leg = ax.legend(loc = "upper right")
leg.get_frame().set_alpha(.5)
ax.set_xlim(left = 0)
outname = "se_imp_site_iw1_imag.pdf"
plt.savefig(outname)
ax.cla()
print outname+" ready"

for fname, c in zip(sys.argv[1:], colors):
    w_max = 10
    sto = Storage(fname)
    g_mom = sto.load("se_imp_iw")
    setup = Setup(g_mom.beta, 0, 0, 0, 0)
    g = setup.transf.backtransform_g(g_mom)
    supermesh = np.array([iw.imag for iw in g.mesh])
    n_iw0 = int(len(supermesh)*.5)
    n_w_max = np.argwhere(supermesh <= w_max)[-1,0]
    mesh = supermesh[n_iw0:n_w_max]
    y = g["up"][0, 0].data[n_iw0:n_w_max,0,0].real
    ax.plot(mesh, y, label = '$\mathrm{'+fname[:-3]+', loc}$', color = c, ls = 'dotted')
    y = g["up"][0, 1].data[n_iw0:n_w_max,0,0].real
    ax.plot(mesh, y, label = '$\mathrm{'+fname[:-3]+', nn}$', color = c)
ax.set_xlabel("$i\\omega_n$")
ax.set_ylabel("$\\Sigma(i\\omega_n)$")
leg = ax.legend(loc = "upper right")
leg.get_frame().set_alpha(.5)
ax.set_xlim(left = 0)
outname = "se_imp_site_iw1_real.pdf"
plt.savefig(outname)
print outname+" ready"

