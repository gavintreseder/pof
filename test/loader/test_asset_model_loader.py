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
from pof.paths import Paths

FILEPATH = Paths().test_path
FILE_NAME_MODEL = Paths().demo_path + r"\Asset Model - Pole - Timber.xlsx"
FILENAME_EXCEL = FILEPATH + r"\Asset Model - Demo.xlsx"
FILENAME_JSON = FILEPATH + r"\Asset Model - Demo.json"


class TestAssetModelLoader(unittest.TestCase):
    def test_class_imports_correctly(self):
        self.assertIsNotNone(AssetModelLoader)

    def test_class_instantiate(self):
        aml = AssetModelLoader()
        self.assertIsNotNone(aml)

    def test_load_failure_mode_loads_and_works_correctly(self):
        """ Load an Asset model and then check a FailureMode can be created from the data and that the object works correctly"""
        aml = AssetModelLoader()
        data_excel = aml.load(FILENAME_EXCEL)
        data_json = aml.load(FILENAME_JSON)

        fm_excel = FailureMode.load(data_excel["pole"]["fm"]["lightning"])
        self.assertIsNotNone(
            fm_excel,
            msg="FailureMode cannot be loaded with excel data from AssetModelLoader",
        )
        fm_json = FailureMode.load(data_json["pole"]["fm"]["lightning"])
        self.assertIsNotNone(
            fm_json,
            msg="FailureMode cannot be loaded with excel data from AssetModelLoader",
        )

        try:
            fm_excel.sim_timeline(200)
        except:
            self.fail(msg="FailureMode cannot sim_timline after excel loaded")
        try:
            fm_json.sim_timeline(200)
        except:
            self.fail(msg="FailureMode cannot sim_timline after json loaded")

        try:
            fm_excel.mc_timeline(t_end=10, n_iterations=10)
        except:
            self.fail(msg="FailureMode cannot sim_timline after excel loaded")
        try:
            fm_json.mc_timeline(t_end=10, n_iterations=10)
        except:
            self.fail(msg="FailureMode cannot sim_timline after json loaded")

    def test_load_component_loads_and_works_correctly(self):
        aml = AssetModelLoader()
        data_excel = aml.load(FILENAME_EXCEL)
        data_json = aml.load(FILENAME_JSON)

        comp_excel = Component.load(data_excel["pole"])
        self.assertIsNotNone(
            comp_excel, msg="Component cannot be loaded with excel data"
        )
        comp_json = Component.load(data_json["pole"])
        self.assertIsNotNone(comp_json, msg="Component cannot be loaded with json data")

        try:
            comp_excel.sim_timeline(200)
        except:
            self.fail(msg="Component cannot sim_timline after excel loaded")
        try:
            comp_json.sim_timeline(200)
        except:
            self.fail(msg="Component cannot sim_timline after json loaded")

        try:
            comp_excel.mc_timeline(t_end=100, n_iterations=10)
        except:
            self.fail(msg="Component cannot mc_timline after excel loaded")
        try:
            comp_json.mc_timeline(t_end=100, n_iterations=10)
        except:
            self.fail(msg="Component cannot mc_timline after json loaded")

        try:
            comp_excel.expected_condition()
        except:
            self.fail(msg="Component cannot get expected_condition after excel loaded")
        try:
            comp_json.expected_condition()
        except:
            self.fail(msg="Component cannot get expected_condition after json loaded")

        try:
            comp_excel.expected_risk_cost_df(t_end=100)
        except:
            self.fail(msg="Component cannot get expected_risk_cost after excel loaded")
        try:
            comp_json.expected_risk_cost_df(t_end=100)
        except:
            self.fail(msg="Component cannot get expected_risk_cost after json loaded")

    def test_failure_mode_indicator(self):
        """ This is a test to investigate the ind & fm issue"""

        aml = AssetModelLoader()
        data = aml.load(FILE_NAME_MODEL)
        comp = Component.load(data["pole"])

        failure_modes_on = ["termites", "fungal decay | internal"]
        tasks_on = [
            "inspection_groundline",
            "functional_failure",
            "conditional_failure",
            "termite_treatment",
        ]

        for fm in comp.fm.values():
            if fm.name not in failure_modes_on:
                fm.active = False
            else:
                fm.active = True

                for task in fm.tasks.values():
                    if task.name in tasks_on:
                        task.active = True
                    else:
                        task.active = False

        # Set up the initial states
        fm = comp.fm["termites"]
        fm.init_states["detection"] = True
        fm.init_states["initiation"] = True
        fm.tasks["inspection_groundline"].t_delay = 0

        comp.mc_timeline(t_end=100, n_iterations=100)

        self.assertEqual(failure_modes_on=1)

    # TODO redo test later
    # def test_load_failure_mode_condition_tasks(self):
    #     aml = AssetModelLoader()
    #     data = aml.load(FILENAME)

    #     fm = FailureMode.load(data["pole"]["fm"]["termites"])

    #     # Change the condition so that the termite powder should be triggered straight away #TODO replace this with asset data
    #     fm.set_init_states({"initiation": True})
    #     fm.set_states({"initiation": True})
    #     for ind in fm.indicators.values():
    #         ind.update_from_dict(
    #             dict(
    #                 pf_interval=100,
    #                 condition=49,
    #             )
    #         )

    #     fm.sim_timeline(400)
    #     fm.plot_timeline()

    #     # Load asset with initiated failure mode, condition already in window and condition task with 100% effectiveness
    #     # self.assertEqual()

    def test_load_lightning_problem(self):

        aml = AssetModelLoader()
        data_excel = aml.load(FILENAME_EXCEL)
        data_json = aml.load(FILENAME_JSON)

        comp_excel = Component.load(data_excel["pole"])
        comp_json = Component.load(data_json["pole"])

        # TODO lightning add a test to make sure the timeline handles before and after correclty
        fm_excel = comp_excel.fm["termites"]
        fm_json = comp_json.fm["termites"]

        fm_excel.sim_timeline(200)
        fm_json.sim_timeline(200)

    def test_mc_timeline_active_False(self):

        # Arrange
        aml = AssetModelLoader()
        data_excel = aml.load(FILENAME_EXCEL)
        comp_excel = Component.load(data_excel["pole"])
        data_json = aml.load(FILENAME_JSON)
        comp_json = Component.load(data_json["pole"])

        for fm in comp_excel.fm.values():
            for task in fm.tasks.values():
                task.update_from_dict({"active": False})

                comp_excel.mc_timeline(t_start=0, t_end=50, n_iterations=10)

            fm.update_from_dict({"active": False})
            comp_excel.mc_timeline(t_start=0, t_end=50, n_iterations=10)

        for fm in comp_json.fm.values():
            for task in fm.tasks.values():
                task.update_from_dict({"active": False})

                comp_json.mc_timeline(t_start=0, t_end=50, n_iterations=10)

            fm.update_from_dict({"active": False})
            comp_json.mc_timeline(t_start=0, t_end=50, n_iterations=10)

        # Assert


if __name__ == "__main__":
    unittest.main()
