"""
    Filename: test_asset_model_loader.py
    Description: Contains the code for testing the AssetModelLoader class
    Author: Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
 
"""

import unittest

import testconfig  # pylint: disable=unused-import
from pof.loader.asset_model_loader import AssetModelLoader
from pof.component import Component
from pof.failure_mode import FailureMode

# TODO move this all to a test file rather than demo
filename = r"C:\Users\gtreseder\OneDrive - KPMG\Documents\3. Client\Essential Energy\Probability of Failure Model\pof\data\inputs\Asset Model - Demo.xlsx"


# TODO move this file into test_loader


class TestAssetModelLoader(unittest.TestCase):
    def test_class_imports_correctly(self):
        self.assertIsNotNone(AssetModelLoader)

    def test_class_instantiate(self):
        aml = AssetModelLoader()
        self.assertIsNotNone(aml)

    def test_load_failure_mode_loads_and_works_correctly(self):
        """ Load an Asset model and then check a FailureMode can be created from the data and that the object works correctly"""
        aml = AssetModelLoader()
        data = aml.load(filename)

        fm = FailureMode.load(data["pole"]["fm"]["termites"])
        self.assertIsNotNone(
            fm, msg="FailureMode cannot be loaded with data from AssetModelLoader"
        )

        try:
            fm.sim_timeline(200)
        except:
            self.fail(msg="FailureMode cannot sim_timline after being loaded")

        try:
            fm.mc_timeline(t_end=10, n_iterations=10)
        except:
            self.fail(msg="FailureMode cannot sim_timline after being loaded")

    def test_load_component_loads_and_works_correctly(self):
        aml = AssetModelLoader()
        data = aml.load(filename)

        comp = Component.load(data["pole"])
        self.assertIsNotNone(comp, msg="Component cannot be loaded with data")

        try:
            comp.sim_timeline(200)
        except:
            self.fail(msg="Component cannot sim_timline after being loaded")

        try:
            comp.mc_timeline(t_end=10, n_iterations=10)
        except:
            self.fail(msg="Component cannot mc_timline after being loaded")

        try:
            comp.expected_condition()
        except:
            self.fail(msg="Component cannot get expected_condition after being loaded")

    def test_load_failure_mode_condition_tasks(self):
        aml = AssetModelLoader()
        data = aml.load(filename)

        fm = FailureMode.load(data["pole"]["fm"]["termites"])

        # Change the condition so that the termite powder should be triggered straight away #TODO replace this with asset data
        fm.set_init_states({"initiation": True})
        fm.set_states({"initiation": True})
        for ind in fm.indicators.values():
            ind.update_from_dict(
                dict(
                    pf_interval=100,
                    condition=49,
                )
            )

        fm.sim_timeline(400)
        fm.plot_timeline()

        # Load asset with initiated failure mode, condition already in window and condition task with 100% effectiveness
        # self.assertEqual()

    def test_load_lightning_problem(self):

        aml = AssetModelLoader()
        data = aml.load(filename)

        comp = Component.load(data["pole"])

        # TODO lightning add a test to make sure the timeline handles before and after correclty
        fm = comp.fm["weathering"]

        fm.sim_timeline(200)


if __name__ == "__main__":
    unittest.main()
