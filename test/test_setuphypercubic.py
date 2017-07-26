import unittest, numpy as np, os, matplotlib as mpl
from pytriqs.gf.local import GfReFreq

from bethe.setups.hypercubic import HypercubicSetup
from bethe.h5interface import Storage
from bethe.selfconsistency import Cycle
from bethe.parameters import TestDMFTParameters
from bethe.plot.cfg import plt, ax


class TestSetupHypercubic(unittest.TestCase):

    def test_init(self):
        setup = HypercubicSetup(15, 1, 2)

    def test_plot_pade(self):
        setup = HypercubicSetup(30, 0, 0, 1, -8, 8, 1000)
        g = setup.gloc
        se = setup.se
        mu = setup.mu
        g.calculate(se, {s:np.array([[mu]]) for s in ['up', 'dn']})
        g_pade = GfReFreq(indices = [0], window = (-12, 12), n_points = 1000)
        g_pade.set_from_pade(g['up'], n_points = 101, freq_offset = np.pi/30.)
        mesh = np.array([w for w in g_pade.mesh])
        ax.plot(mesh.real, -1 / np.pi * g_pade.data[:, 0, 0].imag)
        ax.set_xlabel('$\omega$')
        ax.set_ylabel('$A(\\omega)$')
        ax.set_ylim(bottom = 0)
        plt.savefig('TestSetupHypercubic_test_plot_pade.pdf')
        ax.clear()

    def test_plot_rho_npts_convergence(self):
        nptss = [10,30,100,300,1000]
        n = len(nptss)
        colors = [mpl.cm.viridis(i/float(n-1)) for i in range(n)]
        for npts, col in zip(nptss, colors):
            setup = HypercubicSetup(30, 0, 0, 1, -16, 16, npts)
            g = setup.gloc
            se = setup.se
            mu = setup.mu
            mu = {s:np.array([[mu]]) for s in ['up', 'dn']}
            g.calculate(se, mu)
            n0 = int(len(g.mesh) *.5)
            wmin, wmax = n0, n0+50
            iwmesh = np.array([iw for iw in g.mesh])[wmin:wmax]
            ax.plot(iwmesh.imag, g['up'].data[wmin:wmax, 0, 0].imag, color = col, label = "$"+str(npts)+"$")
        ax.set_xlabel('$\omega$')
        ax.set_ylabel('$\\Im G(i\\omega)$')
        ax.set_xlim(left = 0)
        plt.legend(loc = "lower right", title = "$\\mathrm{npts}$")
        plt.savefig('TestSetupHypercubic_test_plot_rho_npts_convergence.pdf')
        ax.clear()

    def test_plot_rho_w_convergence(self):
        ws = [2,4,8,16]
        n = len(ws)
        colors = [mpl.cm.viridis(i/float(n-1)) for i in range(n)]
        for w, col in zip(ws, colors):
            setup = HypercubicSetup(30, 0, 0, 1, -w, w, 300)
            g = setup.gloc
            se = setup.se
            mu = setup.mu
            mu = {s:np.array([[mu]]) for s in ['up', 'dn']}
            g.calculate(se, mu)
            n0 = int(len(g.mesh) *.5)
            wmin, wmax = n0, n0+50
            iwmesh = np.array([iw for iw in g.mesh])[wmin:wmax]
            ax.plot(iwmesh.imag, g['up'].data[wmin:wmax, 0, 0].imag, color = col, label = "$"+str(w)+"$")
        ax.set_xlabel('$\omega$')
        ax.set_ylabel('$\\Im G(i\\omega)$')
        ax.set_xlim(left = 0)
        plt.legend(loc = "lower right", title = "$\\mathrm{w}$")
        plt.savefig('TestSetupHypercubic_test_plot_rho_w_convergence.pdf')
        ax.clear()

    def test_Cycle_run(self):
        sto = Storage("test.h5")
        params = TestDMFTParameters({'verbosity' : 2, 'measure_g_l': True, 'measure_g_tau': False, 'measure_pert_order': False, 'n_cycles': 10**4, 'n_warmup_cycles': 5*10**3, 'n_l': 30})
        setup = HypercubicSetup(10, 3, 6, -8, 8, 1000)
        cyc = Cycle(sto, params, verbosity = 2, **setup.initialize_cycle())
        cyc.run(1)
        field = .2 * cyc.h_int.sz_tot()
        cyc.h_int = cyc.h_int.get_h_int() + field
        cyc.run(1)
        cyc.h_int -= field
        cyc.run(8)
        #os.remove("test.h5")
