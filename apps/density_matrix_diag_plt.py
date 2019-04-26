import numpy as np, sys

from cdmft.evaluation.generic import Evaluation
from cdmft.h5interface import Storage
from cdmft.plot.cfg import plt, ax


n_bins = 800
omega_max = .8
log = False
offdiag_rows = []
plot_atomic = False
atomic_loop = -1
degeneracy_labels = True
atomic_beta = 1
labels = ["$\\rho_{ii}$"]
for nr in offdiag_rows:
    labels.append("$\\rho_{"+str(nr)+"i}$")
#offdiag_rows = [0]
#offdiag_rows = [36,37,60,61]
#offdiag_rows = [84,1,100]
#offdiag_rows = [38,62,132,136]
for arch in sys.argv[1:]:
    sto = Storage(arch)
    ev = Evaluation(sto)
    rho = ev.get_density_matrix_diag()
    rhorows = [ev.get_density_matrix_row(nr) for nr in offdiag_rows]
    n_plots = 1 + len(rhorows)
    weights = [rho]+ rhorows
    if plot_atomic:
        rho_atom = ev.get_atomic_density_matrix_diag(atomic_loop, atomic_beta)
        n_plots += 1
        weights += [rho_atom]
        labels.append("$\\rho^{atom}_{ii}(\\beta = "+str(atomic_beta)+")$")
    energies = ev.get_energies()
    x = [energies]*n_plots
    if omega_max is None:
        omega_max = rho[0,-1]
    n, bins, patches = ax.hist(x, bins = n_bins, weights = weights, stacked = False, log = log, label = labels)
    if degeneracy_labels:
        bin_degeneracies = np.zeros([len(bins)-1])
        for i in range(len(bins)-1):
            for energy in energies:
                if bins[i] <= energy <= bins[i+1]:
                    bin_degeneracies[i] += 1
        for i, x_bin, bin_deg in zip(range(len(bins)), bins, bin_degeneracies):
            if bin_deg > 0 and x_bin <= omega_max:
                ax.text(x_bin, n[i], str(int(bin_deg)))
    if log:
        ax.set_ylim(bottom = 10**(-4))
    else:
        ax.set_ylim(0,1)
    print np.sort(energies)[:10]
    ax.legend()
    ax.set_ylabel("$\\rho$")
    ax.set_xlabel("$\\omega$")
    ax.set_xlim(right = omega_max)
    outname = arch[:-3]+"_density_matrix_diag.pdf"
    plt.savefig(outname)
    print outname+' ready, note: diag2 recommended'
    plt.close()
