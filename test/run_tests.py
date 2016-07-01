import unittest

from parameters import TestDMFTParameters
from weissfield import TestWeissField
from gfoperations import TestGfOperations
from impuritysolver import TestImpuritySolver
from dmft import TestDMFT
from hamiltonian import TestHamiltonians
from storage import TestLoopStorage
from transformation import TestTransformation


suite = unittest.TestSuite()
suite.addTest(TestDMFTParameters("test_parameters_initialization"))
suite.addTest(TestDMFTParameters("test_parameters_recognization"))
suite.addTest(TestDMFTParameters("test_parameters_interface"))
suite.addTest(TestDMFTParameters("test_parameters_check_for_missing"))
suite.addTest(TestDMFTParameters("test_defaultparameters_initialization"))
suite.addTest(TestWeissField("test_WeissField_initialization"))
suite.addTest(TestWeissField("test_WeissField_selfconsistency"))
suite.addTest(TestWeissField("test_WeissField_set_mu"))
suite.addTest(TestGfOperations("test_sum"))
suite.addTest(TestGfOperations("test_double_dot_product_2by2"))
suite.addTest(TestImpuritySolver("test_ImpuritySolver_initialization"))
suite.addTest(TestImpuritySolver("test_ImpuritySolver_run"))
suite.addTest(TestHamiltonians("test_HubbardSite"))
suite.addTest(TestHamiltonians("test_HubbardPlaquetteMomentum"))
suite.addTest(TestDMFT("test_dmft_initialization"))
suite.addTest(TestDMFT("test_dmft_default_run"))
suite.addTest(TestLoopStorage("test_loopstorage_initialization"))
suite.addTest(TestLoopStorage("test_loopstorage_get_completed_loops"))
suite.addTest(TestTransformation("test_GfStructTransformationIndex"))
suite.addTest(TestTransformation("test_MatrixTransformation"))
suite.addTest(TestTransformation("test_InterfaceToBlockstructure"))

unittest.TextTestRunner(verbosity = 2).run(suite)
