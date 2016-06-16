import unittest

from parameters import TestParameters
from weissfield import TestWeissField
from gfoperations import TestGfOperations
from impuritysolver import TestImpuritySolver

suite = unittest.TestSuite()
suite.addTest(TestParameters("test_parameters_initialization"))
suite.addTest(TestParameters("test_parameters_recognization"))
suite.addTest(TestParameters("test_parameters_interface"))
suite.addTest(TestParameters("test_parameters_check_for_missing"))
suite.addTest(TestWeissField("test_WeissField_initialization"))
suite.addTest(TestWeissField("test_WeissField_selfconsistency"))
suite.addTest(TestWeissField("test_WeissField_set_mu"))
suite.addTest(TestGfOperations("test_sum"))
suite.addTest(TestGfOperations("test_double_dot_product_2by2"))
suite.addTest(TestImpuritySolver("test_ImpuritySolver_initialization"))
suite.addTest(TestImpuritySolver("test_ImpuritySolver_prepare"))
suite.addTest(TestImpuritySolver("test_ImpuritySolver_run"))

unittest.TextTestRunner(verbosity = 2).run(suite)
