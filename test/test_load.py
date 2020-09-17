"""
    Filename: test_load.py
    Description: Contains the code for testing the Constructor class
    Author: Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
 
"""

import unittest
import unittest.mock.patch
import numpy as np

import utils

from pof.load import Load
from pof.config import Config as cf
import pof.demo as demo

"""
class TestLoad(unittest.TestCase):
    @patch(cf.on_error_use_default, False)
    def test_first_case(self):
        self.assertRaises(Load.load, "Invalid Data")

    @patch(cf.on_error_use_default, False)
    def test_n(self):
        ldr = Load()
        ldr = Load("Invalid_data")
        self.assert"""