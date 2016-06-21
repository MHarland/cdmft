import unittest

from Bethe.parameters import DMFTParameters, UnkownParameters, MissingParameters, DefaultDMFTParameters


class TestDMFTParameters(unittest.TestCase):
    
    def test_parameters_initialization(self):
        params = DMFTParameters({"beta": 10, "n_iw": 11, "max_time": 20})

    def test_parameters_recognization(self):
        error_raised = False
        try:
            params = DMFTParameters({"unrecognizable_parameter": 0})
        except UnkownParameters:
            error_raised = True
        self.assertTrue(error_raised)

    def test_parameters_interface(self):
        params = DMFTParameters({"beta": 10})
        for key, val in params:
            if key == "beta":
                self.assertEqual(val, 10)
            else:
                self.assertEqual(val, None)
        self.assertEqual(params["beta"], 10)

    def test_parameters_check_for_missing(self):
        params = DMFTParameters({"n_iw": 10})
        error_raised = False
        try:
            params.assert_setup_complete()
        except MissingParameters:
            error_raised = True
        self.assertTrue(error_raised)
        self.assertEqual(len(params.current) - 1, len(params.missing()))

    def test_defaultparameters_initialization(self):
        p = DefaultDMFTParameters()

    
