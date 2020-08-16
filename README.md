This python code provides a generic (C)DMFT cycle with evaluation tools. It's based on the TRIQS library.

Mandatory Prerequisites:
git@github.com:TRIQS/triqs.git
git@github.com:TRIQS/cthyb.git 

Optional for analytic continuation:
git@github.com:krivenko/som.git
git@bitbucket.org:MHarland/maxent.git

Installation:
Add the cdmft package to PYTHONPATH (e.g. in .bashrc):
export PYTHONPATH="$PYTHONPATH:$PATHTOCDMFT"

Tests:
Change to the test directory and run <pytriqs run_tests.py>

HowTo:
The easy way is to load a provided setup and make it initialize the DMFT-cycle. Example scripts are provided in the example directory. Run e.g. <mpirun -np 2 pytriqs ex_cdmft.py>. For evaluation you can use <pytriqs PATHTOCMDFT/cdmft/apps/SOMESCRIPT.py RESULTSARCHIVE.h5>.