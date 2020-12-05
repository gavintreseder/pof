"""
    Filename: test_condition_task.py
    Description: Contains the code for testing the ConditionTask class
    Author:
        Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
        Illyse Schram  | ischram@kpmg.com.au | illyse.schram@essentialenergy.com.au
"""

import copy
import unittest

import fixtures
import testconfig  # pylint: disable=unused-import
from task.test_task_common import TestTaskCommon
from pof.task import ConditionTask


class TestConditionTask(TestTaskCommon, unittest.TestCase):
    def setUp(self):
        super().setUp()

        # TestTaskCommon Setup
        self._class = ConditionTask
        self._data_valid = [dict(name="TestInspection", task_type="ConditionTask")]
        self._data_invalid_types = [
            {"invalid_input": "invalid_input", "task_type": "ConditionTask"}
        ]
        self._data_complete = copy.deepcopy(fixtures.complete["condition_task"])



if __name__ == "__main__":
    unittest.main()
