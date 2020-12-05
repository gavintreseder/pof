"""
    Filename: test_task.py
    Description: Contains the code for testing the Task class
    Author:
        Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
        Illyse Schram  | ischram@kpmg.com.au | illyse.schram@essentialenergy.com.au
"""

import copy
import unittest

import fixtures
import testconfig  # pylint: disable=unused-import
from task.test_task_common import TestTaskCommon
from pof.task import Task


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
