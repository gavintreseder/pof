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
import pof.demo as demo

#TODO move this all to a test file rather than demo
filename = r"C:\Users\gtreseder\OneDrive - KPMG\Documents\3. Client\Essential Energy\Probability of Failure Model\pof\data\inputs\Asset Model - Demo.xlsx"

#FailureMode().load(data['pole']['fm']['termites'])

#TODO move this file into test_loader

class TestAssetModelLoader(unittest.TestCase):

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

        fm.mc_timeline(t_end = 10, n_iterations = 10)
        self.assertIsNotNone(fm, msg="FailureMode cannot sim_timline after being loaded")

    def test_component_loads(self):
        aml = AssetModelLoader()
        data = aml.load(filename)

        comp = Component.load(data['pole'])
        self.assertIsNotNone(comp, msg="FailureMode cannot be loaded with da")

        #TODO
        self.assertIsNotNone(comp, msg="FailureMode cannot sim_timline after being loaded")

    def test_load_failure_mode_condition_tasks(self):
        aml = AssetModelLoader()
        data = aml.load(filename)

        fm = FailureMode.load(data['pole']['fm']['termites'])
        
        #Change the condition so that the termite powder should be triggered straight away #TODO replace this with asset data
        fm.set_init_state({'initiation':True})
        fm.set_states({'initiation':True})
        for cond in fm.conditions.values():
            cond.update_from_dict(dict(
                pf_interval = 100,
                condition = 49,
            ))


        fm.sim_timeline(400)
        fm.plot_timeline()

        # Load asset with initiated failure mode, condition already in window and condition task with 100% effectiveness
        #self.assertEqual()

    def test_load_component(self):
        aml = AssetModelLoader()
        data = aml.load(filename)

        comp = Component.load(demo.component_data)
        comp2 = Component.load(data['pole'])
        

        comp2.sim_timeline(200)

        # TODO Load asset with initiated failure mode, condition already in window and condition task with 100% effectiveness
        self.assertIsNotNone(comp)



if __name__ == '__main__':
    unittest.main()
