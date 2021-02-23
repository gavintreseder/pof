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
        system.init_timeline(t_end)

        for comp in system.comp.values():
            comp.init_timeline(t_end)
            for fm in comp.fm.values():
                t_fm_timeline_end = fm.timeline["time"][-1]

                self.assertEqual(t_end, t_fm_timeline_end)

    # *************** Test next_tasks ***********************
    def test_next_tasks(self):

        # TODO
        NotImplemented

    def test_next_tasks_one_comp_one_fm_one_task(self):

        t_now = None
        test_next_task = dict(
            pole=(
                5,
                dict(
                    early_life=(20, ["inspection"]),
                    slow_aging=(5, ["inspection"]),
                    fast_aging=(10, ["inspection", "cm"]),
                    random=(15, ["inspection"]),
                ),
            )
        )

        system = System.demo()

        expected = {k: v[1] for k, v in test_next_task.items() if v[0] == 5}

        for comp_name, comp in system.comp.items():
            comp.next_tasks = Mock(return_value=test_next_task[comp_name])
            for fm_name, fm in comp.fm.items():
                fm.next_tasks = Mock(return_value=test_next_task[comp_name][1][fm_name])

        t_next, next_task = system.next_tasks(t_now)

        self.assertEqual(next_task, expected)
        self.assertEqual(t_next, 5)

    def test_next_tasks_many_comp_many_fm_many_task(self):
        """ Mock next tasks """

        #         # TODO new method
        # # Three different task intervals for each of the failure_modes
        # param_next_task = [(5, 5, 5), (5, 5, 5), (10, 5, 5), (10, 10, 5)]

        # for e_l, s_a, f_a, rand in param_next_task:
        #     NotImplemented
        times = dict(
            early_life=[5, 5, 5, 5],
            slow_aging=[5, 5, 5],
            fast_aging=[10, 5, 5],
            random=[10, 10, 5],
        )

        for i in range(3):
            t_now = None
            test_next_task = dict(
                pole=(
                    5,
                    dict(
                        early_life=(times["early_life"][i], ["inspection"]),
                        slow_aging=(times["slow_aging"][i], ["inspection"]),
                        fast_aging=(times["fast_aging"][i], ["inspection", "cm"]),
                        random=(times["random"][i], ["inspection"]),
                    ),
                ),
                other=(
                    10,
                    dict(
                        early_life=(times["early_life"][i], ["inspection"]),
                        slow_aging=(times["slow_aging"][i], ["inspection"]),
                        fast_aging=(times["fast_aging"][i], ["inspection", "cm"]),
                        random=(times["random"][i], ["inspection"]),
                    ),
                ),
            )

            expected = {k: v[1] for k, v in test_next_task.items() if v[0] == 5}

            system = System.demo()

            for comp_name, comp in system.comp.items():
                comp.next_tasks = Mock(return_value=test_next_task[comp_name])
                for fm_name, fm in comp.fm.items():
                    fm.next_tasks = Mock(
                        return_value=test_next_task[comp_name][1][fm_name]
                    )

            t_next, next_task = system.next_tasks(t_now)

            self.assertEqual(next_task, expected)
            self.assertEqual(t_next, 5)

    # *************** Test complete_tasks ***********************

    def test_complete_tasks_ine_comp_one_fm_one_task(self):
        comp_next_tasks = dict(
            pole=dict(
                slow_aging=["inspection"],
            )
        )

        t_now = 5
        system = System.demo()
        system.init_timeline(200)
        system.complete_tasks(t_now, comp_next_tasks)

        for comp_name, comp in system.comp.items():
            for fm_name, fm in comp.fm.items():
                for task_name, task in fm.tasks.items():

                    if comp_name in list(comp_next_tasks):
                        if fm_name in list(comp_next_tasks[comp_name]):
                            if task_name in comp_next_tasks[comp_name][fm_name]:
                                self.assertEqual([t_now], task.t_completion)
                            else:
                                self.assertEqual([], task.t_completion)
                        else:
                            self.assertEqual([], task.t_completion)

    def test_complete_tasks_two_comp_two_fm_two_task(self):

        # Arrange
        comp_next_tasks = dict(
            pole=dict(
                slow_aging=["inspection", "on_condition_replacement"],
                fast_aging=["inspection", "on_condition_replacement"],
            ),
            other=dict(
                slow_aging=["inspection", "on_condition_repair"],
                fast_aging=["inspection", "on_condition_repair"],
            ),
        )
        t_now = 5
        system = System.demo()

        # Act
        with patch.dict("pof.component.cf", {"allow_system_impact": False}):
            system.init_timeline(200)
            system.complete_tasks(t_now, comp_next_tasks)

            # Assert
            for comp_name, comp in system.comp.items():
                for fm_name, fm in comp.fm.items():
                    for task_name, task in fm.tasks.items():
                        if comp_name in list(comp_next_tasks):

                            if fm_name in list(comp_next_tasks[comp_name]):
                                if task_name in comp_next_tasks[comp_name][fm_name]:
                                    self.assertEqual([t_now], task.t_completion)
                                else:
                                    self.assertEqual([], task.t_completion)
                            else:
                                self.assertEqual([], task.t_completion)

    # *************** Test sim_timeline ***********************

    def test_sim_timeline_active_all(self):
        system = System.demo()
        system.sim_timeline(200)

    def test_sim_timeline_active_one(self):
        system = System.demo()

        system.comp["pole"].fm[list(system.comp["pole"].fm)[0]].active = False
        system.sim_timeline(200)

    def test_mp_timeline(self):
        system = System.demo()

        system.mp_timeline(t_end=100)

    def test_mc_timeline(self):
        system = System.demo()

        system.mc_timeline(t_end=100)

    def cancel_sim(self):
        system = System.demo()

        system.mc_timeline(t_end=200)

        system.cancel_sim()

        self.assertEqual(system.up_to_date, False)
        self.assertEqual(system.n, 0)
        self.assertEqual(system.n_sens, 0)

    def test_increment_counter(self):
        system = System.demo()

        system.increment_counter()

        self.assertEqual(system._sim_counter, 1)

    # progress
    # sens progress

    def test_save_load(self):
        system = System.demo()

        file_name = "Test.json"

        system.save(file_name=file_name, file_units="years")

        FILE_NAME_MODEL = Paths().model_path + os.sep + file_name

        aml = AssetModelLoader()
        data = aml.load(FILE_NAME_MODEL)
        system = System.load(data["overhead_network"])

        system.mp_timeline(t_end=100)

    def test_sys_next_tasks(self):

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


if __name__ == "__main__":
    unittest.main()