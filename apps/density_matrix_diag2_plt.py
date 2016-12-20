import matplotlib, numpy as np, sys
matplotlib.use("PDF")
from matplotlib import pyplot as plt
from pytriqs.operators import c as C, c_dag as CDag, n as N

from bethe.hamiltonian import PlaquetteMomentum as Ops
from bethe.h5interface import Storage
from bethe.evaluation.generic import Evaluation
from bethe.evaluation.densitymatrix import StaticObservable


ops = Ops()
n_bins = 400
omega_max = 1
log = False
offdiag_rows = []
plot_atomic = False
atomic_loop = 0
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
    print "loading "+arch+"..."
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
    fig = plt.figure()
    ax = fig.add_axes([.1,.1,.85,.85])
    n, bins, patches = ax.hist(x, bins = n_bins, weights = weights, stacked = False, log = log, label = labels)
    if omega_max is None:
        omega_max = bins[-1]
    if degeneracy_labels:
        n_obs = StaticObservable(ops.n_tot(), sto)
        obs1 = n_obs.get_expectation_value_statewise()
        s2_obs = StaticObservable(ops.ss_tot(), sto)
        obs2 = s2_obs.get_expectation_value_statewise()
        bin_degeneracies = np.zeros([len(bins)])
        bin_obs1 = np.zeros([len(bins)])
        bin_obs2 = np.zeros([len(bins)])
        for i in range(len(bins)-1):
            for j, energy in enumerate(energies):
                if bins[i] <= energy <= bins[i+1]:
                    bin_degeneracies[i] += 1
                    if bin_obs1[i] == 0:
                        bin_obs1[i] = obs1[j]
                    if bin_obs2[i] == 0:
                        bin_obs2[i] = obs2[j]
        for i, x_bin, bin_deg, op1, op2 in zip(range(len(bins)), bins, bin_degeneracies, bin_obs1, bin_obs2):
            if bin_deg > 0.2 and x_bin <= omega_max:
                fontargs = {'color': 'red', 'fontsize': 14}
                #ax.text(x_bin+.005, n[i]-.04, str(int(bin_deg)), **fontargs)
                #ax.text(x_bin+.005, n[i]-.08, str(round(op1,3)), **fontargs)
                #ax.text(x_bin+.005, n[i]-.12, str(round(op2,3)), **fontargs)
                ax.text(x_bin+.005, n[i]+.09, str(int(bin_deg)), **fontargs)
                ax.text(x_bin+.005, n[i]+.05, str(round(op1,3)), **fontargs)
                ax.text(x_bin+.005, n[i]+.01, str(round(op2,3)), **fontargs)
    #ax.text(.01,1-.01,"$g$", transform = ax.transAxes, verticalalignment = "top")
    #ax.text(.01,1-.06,"$N_{tot}$", transform = ax.transAxes, verticalalignment = "top")
    #ax.text(.01,1-.1,"$S_{tot}^2$", transform = ax.transAxes, verticalalignment = "top")
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
    print outname+" ready"
    plt.close()