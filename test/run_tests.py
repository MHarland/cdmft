import unittest

from test_greensfunctions import TestMatsubaraGreensFunction
from test_parameters import TestDMFTParameters
from test_gfoperations import TestGfOperations
from test_impuritysolver import TestImpuritySolver
from test_hubbard import TestHubbard
from test_h5interface import TestStorage
from test_transformation import TestTransformation
from test_schemesgeneric import TestSchemesGeneric
from test_schemesbethe import TestSchemesBethe
from test_selfconsistency import TestCycle
from test_setups import TestSetups
from test_tightbinding import TestTightbinding
from test_schemescdmft import TestSchemesCDMFT


suite = unittest.TestSuite()
suite.addTest(TestMatsubaraGreensFunction("test_MatsubaraGreensFunction_initialization"))
suite.addTest(TestMatsubaraGreensFunction("test_MatsubaraGreensFunction_hdf"))
suite.addTest(TestMatsubaraGreensFunction("test_MatsubaraGreensFunction_fit_tail2"))
#suite.addTest(TestMatsubaraGreensFunction("test_MatsubaraGreensFunction_make_g_tau_real"))
suite.addTest(TestMatsubaraGreensFunction("test_MatsubaraGreensFunction_basic_math"))
suite.addTest(TestDMFTParameters("test_parameters_initialization"))
suite.addTest(TestDMFTParameters("test_parameters_recognization"))
suite.addTest(TestDMFTParameters("test_parameters_interface"))
suite.addTest(TestDMFTParameters("test_parameters_check_for_missing"))
suite.addTest(TestDMFTParameters("test_defaultparameters_initialization"))
suite.addTest(TestGfOperations("test_sum"))
suite.addTest(TestGfOperations("test_double_dot_product_2by2"))
suite.addTest(TestImpuritySolver("test_ImpuritySolver_initialization"))
suite.addTest(TestImpuritySolver("test_ImpuritySolver_run"))
suite.addTest(TestImpuritySolver("test_ImpuritySolver_init_new_giw"))
suite.addTest(TestImpuritySolver("test_ImpuritySolver_get_g_iw"))
suite.addTest(TestHubbard("test_HubbardSite"))
suite.addTest(TestHubbard("test_HubbardPlaquetteMomentum"))
suite.addTest(TestHubbard("test_HubbardPlaquetteMomentumNambu"))
suite.addTest(TestHubbard("test_HubbardTriangleMomentum"))
suite.addTest(TestStorage("test_Storage_initialization"))
suite.addTest(TestStorage("test_Storage_get_completed_loops"))
suite.addTest(TestStorage("test_Storage_save_load_cut_merge"))
suite.addTest(TestTransformation("test_GfStructTransformationIndex"))
suite.addTest(TestTransformation("test_MatrixTransformation"))
suite.addTest(TestTransformation("test_InterfaceToBlockstructure"))
suite.addTest(TestSchemesGeneric("test_SchemesGeneric_inits_and_basic_maths"))
suite.addTest(TestSchemesBethe("test_SchemesBethe_init"))
suite.addTest(TestSchemesBethe("test_SchemesBethe_calculate"))
suite.addTest(TestSchemesBethe("test_SchemesBethe_find_and_set_mu_single"))
#suite.addTest(TestSchemesBethe("test_SchemesBethe_find_and_set_mu_double")) # TODO very unstable
suite.addTest(TestCycle("test_Cycle_initialization"))
#suite.addTest(TestCycle("test_Cycle_run"))
suite.addTest(TestSetups("test_BetheSetups_init"))
suite.addTest(TestTightbinding("test_LatticeDispersion_dimer_in_chain"))
suite.addTest(TestTightbinding("test_LatticeDispersion_dimer_in_chain_transform"))
suite.addTest(TestTightbinding("test_SquarelatticeDispersion"))
#suite.addTest(TestSetups("test_SingleBetheSetup_with_cycle_run"))
suite.addTest(TestSchemesCDMFT("test_SchemesCDMFT_init"))
suite.addTest(TestSchemesCDMFT("test_SchemesCDMFT_dmu"))
suite.addTest(TestSchemesCDMFT("test_SchemesCDMFT_calculate_clustersite_basis"))
suite.addTest(TestSchemesCDMFT("test_SchemesCDMFT_calculate_clustermomentum_basis"))
#suite.addTest(TestSchemesCDMFT("test_SchemesCDMFT_Cycle"))
suite.addTest(TestSetups("test_chain_MomentumDimerCDMFTSetup"))
suite.addTest(TestSetups("test_chain_StrelCDMFTSetup"))
suite.addTest(TestSetups("test_squarelattice_MomentumPlaquetteCDMFTSetup"))
unittest.TextTestRunner(verbosity = 2).run(suite)
