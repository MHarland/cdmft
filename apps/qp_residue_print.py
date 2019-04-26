import sys

from cdmft.evaluation.generic import Evaluation
from cdmft.h5interface import Storage


for fname in sys.argv[1:]:
    sto = Storage(fname)
    ev = Evaluation(sto)
    g = sto.load('g_imp_iw')
    indices = [i for i in g.all_indices]
    print fname+':'
    for i in indices:
        ind_label = i[0]+'_'+i[1]+i[2]
        results = str()
        for n_freqs in range(2,5):
            results += str(ev.get_quasiparticle_residue(n_freqs, i[0], (i[1], i[2])))[:5]+'; '
        print ind_label+': '+results
