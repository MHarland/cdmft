This python code provides a generic DMFT cycle with evaluation tools. It's based on the TRIQS library. So far, it has been written for model-type calculations, i.e. no explicit DFT interface.

Mandatory Prerequisites:
git@github.com:TRIQS/triqs.git
git@github.com:TRIQS/cthyb.git

Optional for analytic continuation:
git@github.com:krivenko/som.git
git@bitbucket.org:MHarland/maxent.git

Installation:
TRIQS is the most demanding installation in terms of linking. Add the bethe (and also the maxent) package to PYTHONPATH.

Tests:
Change to the test directory and run <pytriqs run_tests.py>

HowTo:
The easy way is to load a provided setup and make it initialize the DMFT-cycle. Example scripts are provided in the example directory. Run e.g. <mpirun -np 2 pytriqs ex_cdmft.py>. For evaluation you can use <pytriqs PATHTOBETHE/bethe/apps/SOMESCRIPT.py RESULTSARCHIVE.h5>.