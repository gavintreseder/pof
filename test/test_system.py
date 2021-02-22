import copy
import unittest
from unittest.mock import Mock, patch
import os

from test_pof_base import TestPofBaseCommon
from pof.paths import Paths
from config import config
from pof.system import System
import fixtures
from pof.loader.asset_model_loader import AssetModelLoader

cf = config["System"]


class TestSystem(TestPofBaseCommon, unittest.TestCase):
    """
    Unit tests for the System class including common tests from TestPoFBase
    """

    ## *************** Test setup ***********************

    def setUp(self):
        super().setUp()

        file_path = Paths().test_path + r"\fixtures.py"

        # TestPofBase Setup
        self._class = System
        self._data_valid = [dict(name="TestSystem")]
        self._data_invalid_types = [{"invalid_type": "invalid_type"}]
        self._data_invalid_values = []
        self._data_complete = copy.deepcopy(fixtures.complete["system"])

    def test_class_imports_correctly(self):
        self.assertIsNotNone(System)

    def test_class_instantiate(self):
        system = System()
        self.assertIsNotNone(system)

    ## *************** Test demo ***********************

    def test_demo(self):
        system = System.demo()
        self.assertIsNotNone(system)

    # *************** Test init_timeline ***********************

    def test_init_timeline(self):
        t_end = 200
        system = System.demo()

        for comp in system.comp.values():
            comp.init_timeline(t_end)
            for fm in comp.fm.values():
                t_fm_timeline_end = fm.timeline["time"][-1]

                self.assertEqual(t_end, t_fm_timeline_end)

    # *************** Test sim_timeline ***********************

    def test_sim_timeline_active_all(self):
        system = System.demo()
        system.mp_timeline(200)

    def test_sim_timeline_active_one(self):
        system = System.demo()

        system.comp["pole"].fm[list(system.comp["pole"].fm)[0]].active = False
        system.mp_timeline(200)

    def test_mp_timeline(self):
        system = System.demo()

        system.mp_timeline(t_end=100)

    def test_mc_timeline(self):
        system = System.demo()

        system.mc_timeline(t_end=100)

    # cancel sim
    # increment counter
    # save timeline

    # progress
    # sens progress

    def test_sys_next_tasks(self):

        FILEPATH = Paths().model_path
        FILE_NAME_MODEL = Paths().demo_path + r"\Asset Model.xlsx"

        aml = AssetModelLoader()
        data = aml.load(FILE_NAME_MODEL)
        system = System.load(data["overhead_network"])

        # TODO lightning add a test to make sure the timeline handles before and after correclty
        # fm_excel = sys_excel.comp["pole"].fm["termites"]
        # fm_json = sys_json.comp["pole"].fm["termites"]

        system.mp_timeline(200)

    # ************ Test expected methods *****************

    def test_expected_risk_cost_df(self):  # integration test

        # Arrange
        t_end = 50
        n_iterations = 10
        system = System.demo()

        # Act
        system.mc_timeline(t_end=t_end, n_iterations=n_iterations)
        actual = system.expected_risk_cost_df()

        # Assert
        # TODO make asserts

    # calc_pof_df
    # calc_df_task_forecast
    # calc_df_cond

    def test_expected_condition_with_timelines(self):
        # TODO make it work when mc_timeline hs nto been called

        system = System.demo()
        system.mc_timeline(10)
        system.expected_condition()

    def test_expected_sensitivity(self):

        # Arrange
        t_end = 50
        n_iterations = 2
        sens_var = "overhead_network-comp-pole-task_group_name-groundline-t_interval"
        system = System.demo()

        # Act
        actual = system.expected_sensitivity(var_id=sens_var, lower=1, upper=10)
        # Assert

    # ************ Test reset methods *****************

    def test_reset_condition(self):
        NotImplemented

    def test_reset_for_next_sim(self):

        perfect = 100
        initial = 80
        t_end = 10
        accumulated = abs(perfect - initial)

        system = System.demo()
        for comp in system.comp.values():
            comp.indicator["slow_degrading"].initial = initial

        # Act
        system.mc_timeline(t_end)
        system.reset_for_next_sim()

        # Assert
        for comp in system.comp.values():
            self.assertEqual(
                comp.indicator["slow_degrading"].get_accumulated(), accumulated
            )

    def test_reset(self):

        system = System.demo()
        system.mc_timeline(5)
        system.reset()

        self.assertEqual(system.df_erc, None)
        for comp in system.comp.values():
            self.assertEqual(comp.indicator["slow_degrading"].get_accumulated(), 0)

    # get_objects
    # get_dash_ids
    # get_update_ids

    def test_save(self):
        NotImplemented


if __name__ == "__main__":
    unittest.main()