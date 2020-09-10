"""
    Filename: test_asset_model_loader.py
    Description: Contains the code for testing the AssetModelLoader class
    Author: Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
 
"""

import unittest
import numpy as np

import utils

from pof.loader.asset_model_loader import AssetModelLoader
from pof.component import Component
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


    def test_failure_mode_loads(self):
        aml = AssetModelLoader()
        data = aml.load(filename)

        fm = FailureMode.load(data['pole']['fm']['termites'])
        self.assertIsNotNone(fm, msg="FailureMode cannot be loaded with da")

        fm.sim_timeline(200)
        self.assertIsNotNone(fm, msg="FailureMode cannot sim_timline after being loaded")


    def test_component_loads(self):
        aml = AssetModelLoader()
        data = aml.load(filename)

        comp = Component.load(data['pole'])
        self.assertIsNotNone(comp, msg="FailureMode cannot be loaded with da")

        #TODO
        self.assertIsNotNone(comp, msg="FailureMode cannot sim_timline after being loaded")

    def test_load_component(self):

        """
        FailureMode().load(demo.failure_mode_data['slow_aging'])
        FailureMode().load(data['pole']['fm']['termites'])

        Inspection().load(demo.inspection_data['instant'])
        Inspection().load(data['pole']['fm']['termites']['tasks']['inspection'])

        Component().load(demo.component_data)
        Component().load(data['pole'])
        """
        aml = AssetModelLoader()
        data = aml.load(filename)

        comp = Component()
        comp = comp.load(data['pole'])
        self.assertIsNotNone(comp)


if __name__ == '__main__':
    unittest.main()
