import unittest

from Bethe.parameters import Parameters, UnkownParameters, MissingParameters

class TestParameters(unittest.TestCase):
    
    def test_parameters_initialization(self):
        params = Parameters({"beta": 10, "n_iw": 11, "max_time": 20})

    def test_parameters_recognization(self):
        error_raised = False
        try:
            params = Parameters({"unrecognizable_parameter": 0})
        except UnkownParameters:
            error_raised = True
        self.assertTrue(error_raised)

    def test_parameters_interface(self):
        params = Parameters({"beta": 10})
        for key, val in params:
            self.assertEqual(key, "beta")
            self.assertEqual(val, 10)
        self.assertEqual(params["beta"], 10)

    def test_parameters_check_for_missing(self):
        params = Parameters({"n_iw": 10})
        error_raised = False
        try:
            params.check_for_missing()
        except MissingParameters:
            error_raised = True
        self.assertTrue(error_raised)
        
