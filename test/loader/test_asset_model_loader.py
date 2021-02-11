"""
    Filename: test_asset_model_loader.py
    Description: Contains the code for testing the AssetModelLoader class
    Author: Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
"""

import unittest


import testconfig  # pylint: disable=unused-import
from pof.loader.asset_model_loader import AssetModelLoader
from pof.component import Component
from pof.system import System
from pof.failure_mode import FailureMode
from pof.paths import Paths

FILEPATH = Paths().test_path
FILE_NAME_MODEL = Paths().demo_path + r"\System - Asset Model.xlsx"
FILENAME_EXCEL = FILEPATH + r"\Asset Model - Demo - System.xlsx"
FILENAME_JSON = FILEPATH + r"\Asset Model - Demo - System.json"


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

        fm_excel = FailureMode.load(
            data_excel["overhead_network"]["comp"]["pole"]["fm"]["lightning"]
        )
        self.assertIsNotNone(
            fm_excel,
            msg="FailureMode cannot be loaded with excel data from AssetModelLoader",
        )
        fm_json = FailureMode.load(
            data_json["overhead_network"]["comp"]["pole"]["fm"]["lightning"]
        )
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
        aml_excel = AssetModelLoader(FILENAME_EXCEL)
        data_excel = aml_excel.load(FILENAME_EXCEL)
        aml_json = AssetModelLoader(FILENAME_JSON)
        data_json = aml_json.load(FILENAME_JSON)

        sys_excel = System.load(data_excel["overhead_network"])
        self.assertIsNotNone(sys_excel, msg="System cannot be loaded with excel data")
        sys_json = System.load(data_json["overhead_network"])
        self.assertIsNotNone(sys_json, msg="System cannot be loaded with json data")

        try:
            sys_excel.init_timeline(200)
        except:
            self.fail(msg="System cannot sim_timline after excel loaded")
        try:
            sys_json.init_timeline(200)
        except:
            self.fail(msg="System cannot sim_timline after json loaded")

        try:
            sys_excel.mc_timeline(t_end=100, n_iterations=10)
        except:
            self.fail(msg="System cannot mc_timline after excel loaded")
        try:
            sys_json.mc_timeline(t_end=100, n_iterations=10)
        except:
            self.fail(msg="System cannot mc_timline after json loaded")

        try:
            sys_excel.expected_condition()
        except:
            self.fail(msg="System cannot get expected_condition after excel loaded")
        try:
            sys_json.expected_condition()
        except:
            self.fail(msg="System cannot get expected_condition after json loaded")

        try:
            sys_excel.expected_risk_cost_df(t_end=100)
        except:
            self.fail(msg="System cannot get expected_risk_cost after excel loaded")
        try:
            sys_json.expected_risk_cost_df(t_end=100)
        except:
            self.fail(msg="System cannot get expected_risk_cost after json loaded")

    def test_failure_mode_indicator(self):
        """ This is a test to investigate the ind & fm issue"""

        aml = AssetModelLoader(FILE_NAME_MODEL)
        data = aml.load(FILE_NAME_MODEL)
        sys = System.from_dict(data["overhead_network"])

        failure_modes_on = ["termites", "fungal decay | internal"]
        tasks_on = [
            "inspection_groundline",
            "functional_failure",
            "conditional_failure",
            "termite_treatment",
        ]

        for comp in sys.comp.values():
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

        sys.mc_timeline(t_end=100, n_iterations=100)

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

        sys_excel = System.load(data_excel["overhead_network"])
        sys_json = System.load(data_json["overhead_network"])

        # TODO lightning add a test to make sure the timeline handles before and after correclty
        fm_excel = sys_excel.comp["pole"].fm["termites"]
        fm_json = sys_json.comp["pole"].fm["termites"]

        fm_excel.sim_timeline(200)
        fm_json.sim_timeline(200)

    def test_mc_timeline_active_False(self):

        # Arrange
        aml = AssetModelLoader()
        data_excel = aml.load(FILENAME_EXCEL)
        sys_excel = System.load(data_excel["overhead_network"])
        data_json = aml.load(FILENAME_JSON)
        sys_json = System.load(data_json["overhead_network"])

        for comp in sys_excel.comp.values():
            for fm in comp.fm.values():
                for task in fm.tasks.values():
                    task.update_from_dict({"active": False})

                    sys_excel.mc_timeline(t_start=0, t_end=50, n_iterations=10)

                fm.update_from_dict({"active": False})
                sys_excel.mc_timeline(t_start=0, t_end=50, n_iterations=10)

            comp.update_from_dict({"active": False})
            sys_excel.mc_timeline(t_start=0, t_end=50, n_iterations=10)

        for comp in sys_json.comp.values():
            for fm in comp.fm.values():
                for task in fm.tasks.values():
                    task.update_from_dict({"active": False})

                    sys_json.mc_timeline(t_start=0, t_end=50, n_iterations=10)

                fm.update_from_dict({"active": False})
                sys_json.mc_timeline(t_start=0, t_end=50, n_iterations=10)

            comp.update_from_dict({"active": False})
            sys_json.mc_timeline(t_start=0, t_end=50, n_iterations=10)

        # Assert


if __name__ == "__main__":
    unittest.main()
