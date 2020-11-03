"""
    Filename: test_component.py
    Description: Contains the code for testing the Component class
    Author: Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
 
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import scipy.stats as ss

from test_load import TestPofBase
import fixtures
import testconfig
from pof.component import Component
from config import config
import pof.demo as demo

from pof.interface.figures import update_condition_fig

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
        self._data_complete = [fixtures.complete['component_0'], fixtures.complete['component_0']]

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

    def test_next_tasks_many_fm_many_tasks_new_method(self):
        # TODO new method
        # Three different task intervals for each of the failure_modes
        param_next_task = [(5, 5, 5), (5, 5, 5), (10, 5, 5), (10, 10, 5)]

        for e_l, s_a, f_a, rand in param_next_task:
            NotImplemented

    def test_next_tasks_many_fm_many_task(self):
        """ Mock next tasks """
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

    def test_mc_timeline(self):
        comp = Component.demo()

        comp.mc_timeline(t_end=100)

    # ************ Test expected methods *****************

    def test_expected_condition_no_timeline(self):
        comp = Component.demo()
        comp.expected_condition()

        # TODO add some checks

    def test_expected_condition_with_timelines(self):
        comp = Component.demo()
        comp.mc_timeline(10)
        comp.expected_condition()

    # ************ Test update methods *****************

    def test_update(self):
        # TODO test all values
        comp = Component.demo()

        comp.update("comp-fm-slow_aging-active", False)
        self.assertEqual(comp.fm["slow_aging"].active, False)

    def test_update_from_str(self):

        expected_list = [True]

        comp = Component.demo()
        dash_ids = comp.get_dash_ids()

        for dash_id in dash_ids:

            for expected in expected_list:

                comp.update(dash_id, expected)

                val = NotImplemented

                self.assertEqual(val, expected, msg="Error: dash_id %s" % (dash_id))

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

        comp = Component.demo()
        comp.indicator["slow_degrading"].set_initial(20)

        comp.mc_timeline(10)
        comp.reset()

        self.assertEqual(comp.indicator["slow_degrading"].get_accumulated(), 20)

    def test_replace(self):
        NotImplemented


if __name__ == "__main__":
    unittest.main()
