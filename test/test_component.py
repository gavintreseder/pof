"""
    Filename: test_component.py
    Description: Contains the code for testing the Component class
    Author: Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
"""

import unittest
from unittest.mock import Mock, patch

from test_load import TestPofBase
import fixtures
import testconfig  # pylint: disable=unused-import
from pof.component import Component
from config import config

cf = config["Component"]


class TestComponent(TestPofBase, unittest.TestCase):
    """
    Unit tests for the Component class
    """

    def setUp(self):
        super().setUp()

        # TestPofBase Setup
        self._class = Component
        self._data_valid = [dict(name="TestComponent")]
        self._data_invalid_types = [{"invalid_type": "invalid_type"}]
        self._data_invalid_values = []
        self._data_complete = [
            fixtures.complete["component_0"],
            fixtures.complete["component_0"],
        ]

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
            max_risk = comp.fm["random"].consequence.risk_cost_total

            # Assert
            self.assertLessEqual(risk, max_risk)

    # ************ Test expected methods *****************

    def test_expected_condition_with_timelines(self):
        # TODO make it work when mc_timeline hs nto been called

        comp = Component.demo()
        comp.mc_timeline(10)
        comp.expected_condition()

    # ************ Test update methods *****************

    def test_update(self):
        # TODO test all values
        comp = Component.demo()

        comp.update("comp-fm-slow_aging-active", False)
        self.assertEqual(comp.fm["slow_aging"].active, False)

    def test_expected_inspection_interval(self):

        NotImplemented

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


if __name__ == "__main__":
    unittest.main()
