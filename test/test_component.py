"""
    Filename: test_component.py
    Description: Contains the code for testing the Component class
    Author: Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
"""

import copy
import unittest
from unittest.mock import Mock, patch
import os

from test_pof_base import TestPofBaseCommon
from pof.paths import Paths
import testconfig  # pylint: disable=unused-import
from pof.component import Component, calc_confidence_interval, sort_df
from config import config
from pof.interface.figures import calc_y_max
from pof.data.asset_data import SimpleFleet
import fixtures
from pof.units import scale_units
from pof.component import Component

cf = config["Component"]


class TestComponent(TestPofBaseCommon, unittest.TestCase):
    """
    Unit tests for the Component class incldding common tests from TestPoFBase
    """

    def setUp(self):
        super().setUp()

        file_path = Paths().test_path + r"\fixtures.py"

        # TestPofBase Setup
        self._class = Component
        self._data_valid = [dict(name="TestComponent")]
        self._data_invalid_types = [{"invalid_type": "invalid_type"}]
        self._data_invalid_values = []
        self._data_complete = copy.deepcopy(fixtures.complete["component"])

    def test_class_imports_correctly(self):
        self.assertIsNotNone(Component)

    def test_class_instantiate(self):
        comp = Component()
        self.assertIsNotNone(comp)

    ## *************** Test demo ***********************

    def test_demo(self):
        comp = Component.demo()
        self.assertIsNotNone(comp)

    # *************** Test init_timeline ***********************

    def test_init_timeline(self):
        t_end = 200
        comp = Component.demo()
        comp.init_timeline(t_end)

        for fm in comp.fm.values():
            t_fm_timeline_end = fm.timeline["time"][-1]

            self.assertEqual(t_end, t_fm_timeline_end)

    # *************** Test complete_tasks ***********************

    def test_complete_tasks_one_fm_one_task(self):
        fm_next_tasks = dict(
            slow_aging=["inspection"],
        )
        t_now = 5
        comp = Component.demo()
        comp.init_timeline(200)
        comp.complete_tasks(t_now, fm_next_tasks)

        for fm_name, fm in comp.fm.items():
            for task_name, task in fm.tasks.items():

                if fm_name in list(fm_next_tasks):
                    if task_name in fm_next_tasks[fm_name]:
                        self.assertEqual([t_now], task.t_completion)
                    else:
                        self.assertEqual([], task.t_completion)
                else:
                    self.assertEqual([], task.t_completion)

    def test_complete_tasks_two_fm_two_task(self):

        # Arrange
        fm_next_tasks = dict(
            slow_aging=["inspection", "on_condition_replacement"],
            fast_aging=["inspection", "on_condition_replacement"],
        )
        t_now = 5
        comp = Component.demo()

        # Act
        with patch.dict("pof.component.cf", {"allow_system_impact": False}):
            comp.init_timeline(200)
            comp.complete_tasks(t_now, fm_next_tasks)

            # Assert
            for fm_name, fm in comp.fm.items():
                for task_name, task in fm.tasks.items():

                    if fm_name in list(fm_next_tasks):
                        if task_name in fm_next_tasks[fm_name]:
                            self.assertEqual([t_now], task.t_completion)
                        else:
                            self.assertEqual([], task.t_completion)
                    else:
                        self.assertEqual([], task.t_completion)

    # *************** Test next_tasks ***********************

    def test_next_tasks(self):

        # TODO
        NotImplemented

    def test_next_tasks_one_fm_one_task(self):

        t_now = None
        test_next_task = dict(
            early_life=(20, ["inspection"]),
            slow_aging=(5, ["inspection"]),
            fast_aging=(10, ["inspection", "cm"]),
            random=(15, ["inspection"]),
        )

        expected = {k: v[1] for k, v in test_next_task.items() if v[0] == 5}

        comp = Component.demo()

        for fm_name, fm in comp.fm.items():
            fm.next_tasks = Mock(return_value=test_next_task[fm_name])

        t_next, next_task = comp.next_tasks(t_now)

        self.assertEqual(next_task, expected)
        self.assertEqual(t_next, 5)

    def test_next_tasks_many_fm_many_task(self):
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
                early_life=(times["early_life"][i], ["inspection"]),
                slow_aging=(times["slow_aging"][i], ["inspection"]),
                fast_aging=(times["fast_aging"][i], ["inspection", "cm"]),
                random=(times["random"][i], ["inspection"]),
            )

            expected = {k: v[1] for k, v in test_next_task.items() if v[0] == 5}

            comp = Component.demo()

            for fm_name, fm in comp.fm.items():
                fm.next_tasks = Mock(return_value=test_next_task[fm_name])

            t_next, next_task = comp.next_tasks(t_now)

            self.assertEqual(next_task, expected)
            self.assertEqual(t_next, 5)

    # *************** Test sim_timeline ***********************

    def test_sim_timeline_active_all(self):
        comp = Component.demo()
        comp.sim_timeline(200)

    def test_sim_timeline_active_one(self):
        comp = Component.demo()

        comp.fm[list(comp.fm)[0]].active = False
        comp.sim_timeline(200)

    def test_mp_timeline(self):
        comp = Component.demo()

        comp.mp_timeline(t_end=100)

    def test_mc_timeline(self):
        comp = Component.demo()

        comp.mc_timeline(t_end=100)

    def test_mc_timeline_remain_failed(self):
        """ Check that only one failure mode is triggered when remain failed is true"""
        # Arrange
        comp = Component.demo()

        t_end = 2000

        with patch.dict(
            "pof.component.config",
            {
                "FailureMode": {"remain_failed": True},
                "Component": {"allow_system_impact": True},
            },
        ):

            # Act
            comp.mc_timeline(t_end=t_end, n_iterations=10)
            df = comp.expected_risk_cost_df()
            risk = df.loc[df["task"] == "risk"]["cost"].sum()
            max_risk = comp.fm["random"].consequence.cost

            # Assert
            self.assertLessEqual(risk, max_risk)

    # ************ Test expected methods *****************

    def test_expected_sensitivity(self):

        # Arrange
        t_end = 50
        n_iterations = 2
        sens_var = ""
        comp = Component.demo()

        # Act
        actual = comp.expected_sensitivity(var_id=sens_var, lower=1, upper=10)
        # Assert

    def test_expected_risk_cost_df(self):  # integration test

        # Arrange
        t_end = 50
        n_iterations = 10
        comp = Component.demo()

        # Act
        comp.mc_timeline(t_end=t_end, n_iterations=n_iterations)
        actual = comp.expected_risk_cost_df()

        # Assert
        # TODO make asserts

    def test_expected_condition_with_timelines(self):
        # TODO make it work when mc_timeline hs nto been called

        comp = Component.demo()
        comp.mc_timeline(10)
        comp.expected_condition()

    def test_expected_inspection_interval(self):

        NotImplemented

    # ************ Test Update for Task Group *********

    def test_update_with_task_group(self):
        """
        Check that update_task_group
        """

        attr = "trigger"
        before = "before_update"
        after = "after_update"

        to_change = "group_1"
        not_to_change = "group_2"

        # Arrange
        task_1 = {"name": "task_1", "task_group_name": to_change, attr: before}
        task_2 = {"name": "task_2", "task_group_name": to_change, attr: before}
        task_3 = {"name": "task_3", "task_group_name": not_to_change, attr: before}

        tasks = {"task_1": task_1, "task_2": task_2, "task_3": task_3}

        fm_1 = {"name": "fm_1", "tasks": tasks}
        fm_2 = {"name": "fm_2", "tasks": tasks}

        comp = Component(fm={"fm_1": fm_1, "fm_2": fm_2})

        # update = {"comp": {"task_group_name": {to_change: {attr: {after}}}}}
        update_str = f"comp-task_group_name-{to_change}-{attr}"

        # Act
        comp.update(update_str, after)

        # Assert
        for fm in comp.fm.values():
            for task in fm.tasks.values():

                actual = getattr(task, attr)
                if task.task_group_name == to_change:
                    expected = after
                else:
                    expected = before

                self.assertEquals(actual, expected)

    # ************ Test reset methods *****************

    def test_reset(self):

        comp = Component.demo()
        comp.mc_timeline(5)
        comp.reset()

        self.assertEqual(comp._sim_counter, 0)
        self.assertEqual(comp.indicator["slow_degrading"].get_accumulated(), 0)

    def test_reset_for_next_sim(self):

        perfect = 100
        initial = 80
        t_end = 10
        accumulated = abs(perfect - initial)

        comp = Component.demo()
        comp.indicator["slow_degrading"].initial = initial

        # Act
        comp.mc_timeline(t_end)
        comp.reset_for_next_sim()

        # Assert
        self.assertEqual(
            comp.indicator["slow_degrading"].get_accumulated(), accumulated
        )

    def test_replace(self):
        NotImplemented

    # ************ Test summary methods *************

    def test_calc_confidence_interval(self):
        comp = Component.demo()
        comp.mc_timeline(t_end=100)

        # Arrange
        total_failed = 10

        # Forecast years
        START_YEAR = 2015
        END_YEAR = 2024
        CURRENT_YEAR = 2020

        paths = Paths()

        # Population Data
        file_path = paths.input_path + os.sep
        FILE_NAME = r"population_summary.csv"

        sfd = SimpleFleet(file_path + FILE_NAME)
        sfd.load()
        sfd.calc_age_forecast(START_YEAR, END_YEAR, CURRENT_YEAR)

        df_cohort = sfd.df_age

        # Act
        lower_bound, upper_bound = calc_confidence_interval(
            sim_counter=10, df_cohort=df_cohort, total_failed=total_failed
        )

        # Assert
        self.assertAlmostEqual(total_failed - lower_bound, upper_bound - total_failed)

    def test_sort_df_task(self):
        ##### TEST 1 - TASK #####
        # Arrange
        t_end = 100
        comp = Component.demo()
        comp.mc_timeline(t_end=t_end)

        df_task = comp.expected_risk_cost_df(t_start=0, t_end=t_end)

        # Act
        df_sorted_task = sort_df(df=df_task, column="task")

        # Assert - should be sorted by time then task
        time_vals_task = df_sorted_task["time"].unique()
        task_vals = df_sorted_task["task"].unique()

        for i in range(0, len(time_vals_task) - 1):
            self.assertGreater(time_vals_task[i + 1], time_vals_task[i])

        self.assertEqual(task_vals[0], "risk")

    def test_sort_df_sens(self):
        ##### TEST 2 - SENS #####
        # Arrange
        t_end = 100
        comp = Component.demo()
        comp.mc_timeline(t_end=t_end)

        comp.expected_sensitivity(
            var_id="overhead_network-comp-pole-task_group_name-groundline-t_interval",
            lower=0,
            upper=10,
        )
        df_source = comp.sens_summary(var_name="t_interval")

        # Act
        df_sorted_source = sort_df(
            df=df_source,
            column="source",
            var="t_interval",
        )

        # Assert - should be sorted by time then task
        time_vals_source = df_sorted_source["t_interval"].unique()
        source_vals = df_sorted_source["source"].unique()

        for i in range(0, len(time_vals_source) - 1):
            self.assertGreater(time_vals_source[i + 1], time_vals_source[i])

        self.assertEqual(source_vals[0], "total")
        self.assertEqual(source_vals[1], "risk")
        self.assertEqual(source_vals[2], "direct")

    # ************* Test charts *********************

    def test_calc_y_max(self):
        comp = Component.demo()

        comp.mp_timeline(t_end=200, n_iterations=10)

        comp.expected_risk_cost_df(t_end=200)

        prev_ms_cost = comp.plot_ms(y_axis="cost", keep_axis=True, prev=None)
        prev_ms_cumulative = comp.plot_ms(
            y_axis="cost_cumulative", keep_axis=True, prev=None
        )

        y_max_ms_cost = calc_y_max(
            keep_axis=True, method="max", prev=prev_ms_cost, test=True
        )
        y_max_ms_cumulative = calc_y_max(
            keep_axis=True, method="sum", prev=prev_ms_cumulative, test=True
        )

        df = comp.df_erc

        y_max_ms_cost_df = df["cost"].max() * 1.05

        y_max_ms_cumulative_df = (
            df.groupby("time")["cost_cumulative"].sum().max() * 1.05
        )

        self.assertEqual(y_max_ms_cost, y_max_ms_cost_df)
        self.assertEqual(y_max_ms_cumulative, y_max_ms_cumulative_df)


if __name__ == "__main__":
    unittest.main()
