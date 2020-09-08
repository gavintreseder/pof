"""
    Filename: test_asset_model_loader.py
    Description: Contains the code for testing the AssetModelLoader class
    Author: Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
 
"""

import unittest
import numpy as np

import utils

from pof.loader.asset_model_loader import AssetModelLoader
from pof.failure_mode import FailureMode

filename = r"C:\Users\gtreseder\OneDrive - KPMG\Documents\3. Client\Essential Energy\Probability of Failure Model\pof\data\inputs\Asset Model - Demo.xlsx"

#FailureMode().load(data['pole']['fm']['termites'])

#TODO move this file into test_loader

class TestTask(unittest.TestCase):

    def test_imports_correctly(self):
        self.assertTrue(True)

    def test_instantiate(self):
        aml = AssetModelLoader()
        self.assertIsNotNone(aml)


    def test_valid_failure_mode(self):
        aml = AssetModelLoader()
        data = aml.load(filename)

        fm = FailureMode()
        fm.load(data['pole']['fm']['termites'])
        
        self.assertIsNotNone(fm)

if __name__ == '__main__':
    unittest.main()
