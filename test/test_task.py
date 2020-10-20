"""
    Filename: test_task.py
    Description: Contains the code for testing the Task class
    Author:
        Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
        Illyse Schram  | ischram@kpmg.com.au | illyse.schram@essentialenergy.com.au
"""

import copy
import unittest
from unittest.mock import Mock

import numpy as np

import fixtures
import testconfig
from test_load import TestPofBase
from pof.task import Task, ScheduledTask, ConditionTask, Inspection


class TestTaskCommon(TestPofBase):
    """
    A base class for tests that are expected to work with all Task objects
    """

    def setUp(self):

        super().setUp()

        # TestPofBase
        self._class = self._class
        self._data_valid = self._data_valid
        self._data_invalid_types = ["string", True]
        self._data_invalid_types = [dict(invalid_input="invalid_input")]
        self._data_invalid_values = []


class TestTask(TestTaskCommon, unittest.TestCase):
    def setUp(self):
        super().setUp()

        self._class = Task
        self._data_valid = dict(name="TaskTest", activity="Task")

    # **************** test_load ***********************

    def test_load_empty(self):
        task = Task.load()
        self.assertIsNotNone(task)

    def test_load_valid_dict(self):
        task = Task.load(fixtures.inspection_data["instant"])
        self.assertIsNotNone(task)

    # **************** test_update ***********************

    def test_update(self):

        test_data_1 = copy.deepcopy(fixtures.on_condition_replacement_data)
        test_data_1["cost"] = 0
        test_data_1["triggers"]["condition"]["fast_degrading"]["upper"] = 90
        test_data_2 = copy.deepcopy(fixtures.on_condition_replacement_data)

        # Test all the options
        t1 = Task.from_dict(test_data_1)
        t2 = Task.from_dict(test_data_2)

        t1.update_from_dict(
            {"cost": 5000, "trigger": {"condition": {"fast_degrading": {"upper": 20}}}}
        )

        # self.assertEqual(t1.__dict__, t2.__dict__)
        self.assertEqual(t1.cost, t2.cost)
        self.assertEqual(t1.triggers, t2.triggers)

    def test_update_error(self):

        test_data = copy.deepcopy(fixtures.on_condition_replacement_data)

        t = Task.from_dict(test_data)

        update = {"alpha": 10, "beta": 5}

        self.assertRaises(KeyError, t.update_from_dict, update)


class TestScheduledTask(TestTaskCommon, unittest.TestCase):
    """
    Tests the functionality of the ScheduledTask class including all common tests written in TestTaskCommon
    """

    def setUp(self):
        super().setUp()

        self._class = ScheduledTask
        self._data_valid = dict(name="ScheduledTaskTest", activity="ScheduledTask")

    def test_sim_timeline(self):
        """Check the a scheduled task returns the correct time"""
        # Task params
        param_t_delay = [0, 1, 5]
        param_t_interval = [0, 1, 5]

        # Sim_timeline params
        param_inputs = [(0, 100)]

        for t_interval in param_t_interval:
            for t_delay in param_t_delay:

                for t_start, t_end in param_inputs:

                    # with self.subTest():
                    # Arrange
                    task = ScheduledTask(t_delay=t_delay, t_interval=t_interval)

                    if t_delay == 0:
                        delay = []
                    else:
                        delay = np.linspace(t_interval, 0, t_delay + 1)

                    expected = np.concatenate(
                        [
                            delay,
                            np.tile(
                                np.linspace(t_interval, 0, t_interval + 1),
                                int((t_end - t_delay) / t_interval) + 1,
                            ),
                        ]
                    )

                    expected = expected[t_start : t_end + 1]

                    # Act
                    schedule = task.sim_timeline(t_start=t_start, t_end=t_end)

                    # Assert
                    np.testing.assert_array_equal(expected, schedule)


class TestConditionTask(TestTaskCommon, unittest.TestCase):
    def setUp(self):
        super().setUp()

        # TestTaskCommon Setup
        self._class = ConditionTask
        self._data_valid = dict(name="TestInspection", activity="ConditionTask")

    # **************** test_load ***********************

    def test_load_empty(self):
        task = ConditionTask.load()
        self.assertIsNotNone(task)

    def test_load_valid_dict(self):
        task = ConditionTask.load(fixtures.on_condition_replacement_data)
        self.assertIsNotNone(task)

    def test_update(self):

        test_data_1 = copy.deepcopy(fixtures.on_condition_replacement_data)
        test_data_1["cost"] = 0
        test_data_1["triggers"]["condition"]["fast_degrading"]["upper"] = 90
        test_data_2 = copy.deepcopy(fixtures.on_condition_replacement_data)

        # Test all the options
        t1 = ConditionTask.from_dict(test_data_1)
        t2 = ConditionTask.from_dict(test_data_2)

        t1.update_from_dict(
            {"cost": 5000, "trigger": {"condition": {"fast_degrading": {"upper": 20}}}}
        )

        # self.assertEqual(t1.__dict__, t2.__dict__)
        self.assertEqual(t1.cost, t2.cost)
        self.assertEqual(t1.triggers, t2.triggers)

    def test_update_error(self):

        test_data = copy.deepcopy(fixtures.on_condition_replacement_data)

        t = ConditionTask.from_dict(test_data)

        update = {"alpha": 10, "beta": 5}

        self.assertRaises(KeyError, t.update_from_dict, update)


class TestInspection(TestTaskCommon, unittest.TestCase):
    def setUp(self):
        super().setUp()

        # TestTaskCommon Setup
        self._class = Inspection
        self._data_valid = dict(name="TestInspection", activity="Inspection")

    # **************** test_load ***********************

    def test_load_valid_dict(self):
        task = Inspection.load(fixtures.inspection_data["instant"])
        self.assertIsNotNone(task)

    # **************** test_update ***********************

    def test_update(self):

        test_data_1 = copy.deepcopy(fixtures.on_condition_replacement_data)
        test_data_1["cost"] = 0
        test_data_1["triggers"]["condition"]["fast_degrading"]["upper"] = 90
        test_data_2 = copy.deepcopy(fixtures.on_condition_replacement_data)

        # Test all the options
        t1 = ConditionTask.from_dict(test_data_1)
        t2 = ConditionTask.from_dict(test_data_2)

        t1.update_from_dict(
            {"cost": 5000, "trigger": {"condition": {"fast_degrading": {"upper": 20}}}}
        )

        # self.assertEqual(t1.__dict__, t2.__dict__)
        self.assertEqual(t1.cost, t2.cost)
        self.assertEqual(t1.triggers, t2.triggers)

    def test_update_error(self):

        test_data = copy.deepcopy(fixtures.inspection_data["instant"])

        t = Inspection.from_dict(test_data)

        update = {"alpha": 10, "beta": 5}

        self.assertRaises(KeyError, t.update_from_dict, update)


if __name__ == "__main__":
    unittest.main()
