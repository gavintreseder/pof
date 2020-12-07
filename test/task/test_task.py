"""
    Filename: test_task.py
    Description: Contains the code for testing the Task class
    Author:
        Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
        Illyse Schram  | ischram@kpmg.com.au | illyse.schram@essentialenergy.com.au
"""

import copy
import unittest

import numpy as np

import fixtures
import testconfig  # pylint: disable=unused-import
from test_pof_base import TestPofBaseCommon
from pof.task import Task


class TestTaskCommon(TestPofBaseCommon):
    """
    A base class for tests that are expected to work with all Task objects
    """

    def setUp(self):

        super().setUp()

        # TestPofBase
        # Overide in all children classes
        # self._class
        # self._valid
        # self._invalid_types
        self._data_invalid_values = []
        # self._data_complete

    def test_sim_timeline_active_false(self):

        # Arrange
        t_start = 0
        t_end = 50
        t_range = t_end - t_start + 1
        timeline = {
            "time": np.linspace(t_start, t_end, t_end + 1, dtype=int),
            "initiation": np.full(t_range, False),
            "detection": np.full(t_range, False),
            "failure": np.full(t_range, False),
        }

        task = self._class.from_dict(self._data_complete[0])
        task.active = False

        expected = np.full(t_range, -1)  # 51

        # Act
        actual = task.sim_timeline(t_start=t_start, t_end=t_end, timeline=timeline)

        # Assert
        np.testing.assert_array_equal(expected, actual)


class TestTask(TestTaskCommon, unittest.TestCase):
    """
    Tests the functionality of the Task class including all common tests written in TestTaskCommon
    """

    def setUp(self):
        super().setUp()

        self._class = Task
        self._data_valid = [dict(name="TaskTest", task_type="Task")]
        self._data_invalid_types = [
            dict(invalid_input="invalid_input", task_type="Task")
        ]
        self._data_complete = copy.deepcopy(fixtures.complete["task"])


if __name__ == "__main__":
    unittest.main()
