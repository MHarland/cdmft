import matplotlib, sys, numpy as np
matplotlib.use("PDF")
from matplotlib import pyplot as plt

from cdmft.h5interface import Storage
from cdmft.setups.cdmftsquarelattice import MomentumPlaquetteSetup
from cdmft.transformation import MatrixTransformation


sto = Storage('ex_cdmft.h5')
beta = 10
setup = MomentumPlaquetteSetup(beta, 2, 4, -1, 0, 16)
site_to_mom = MatrixTransformation(setup.old_struct, setup.site_transf_mat, setup.new_struct)
g_iw_mom = sto.load('g_loc_iw')
g = site_to_mom.backtransform_g(g_iw_mom)

fig = plt.figure()
ax = fig.add_axes([.12,.12,.85,.85])
w_max = 10
supermesh = np.array([iw.imag for iw in g.mesh])
n_iw0 = int(len(supermesh)*.5)
n_w_max = np.argwhere(supermesh <= w_max)[-1,0]
mesh = supermesh[n_iw0:n_w_max]
orbs = [('up',0,0), ('up',0,1), ('up',0,3)]
n_orbs = len(orbs)
colors = [matplotlib.cm.jet(i/float(max(1,n_orbs-1))) for i in range(n_orbs)]
for orb, c in zip(orbs, colors):
    b, i, j = orb
    y = g[b][i, j].data[n_iw0:n_w_max,0,0].imag
    ax.plot(mesh, y, label = b+str(i)+str(j), color = c)
    y = g[b][i, j].data[n_iw0:n_w_max,0,0].real
    ax.plot(mesh, y, color = c, ls = '--')
ax.set_xlabel("$i\\omega_n$")
ax.set_ylabel("$G(i\\omega_n)$")
ax.legend(fontsize = 8, loc = "upper right")
ax.set_xlim(left = 0)
outname = "g_imp_iw.pdf"
plt.savefig(outname)
print outname+" ready"
plt.close()
