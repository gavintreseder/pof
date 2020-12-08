"""
    Filename: test_condition_task.py
    Description: Contains the code for testing the ConditionTask class
    Author:
        Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
        Illyse Schram  | ischram@kpmg.com.au | illyse.schram@essentialenergy.com.au
"""

import copy
import unittest
from unittest.mock import Mock

import numpy as np

import fixtures
import testconfig  # pylint: disable=unused-import
from .test_task import TestTaskCommon
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

    def test_sim_timeline(self):

        # Arrange
        t_end = 100

        condition = np.linspace(t_end, 0, t_end + 1)
        t = len(condition)

        timeline = {
            "initiation": np.full(t, True),
            "detection": np.full(t, True),
            "failure": np.full(t, False),
        }

        indicator = Mock()
        indicator.get_timeline = Mock(return_value=condition)

        indicators = {"condition_1": indicator, "condition_2": indicator}

        param_list = [()]

        expected = np.array([-1] * 10 + [0] * 91)

        task = ConditionTask(
            triggers={
                "state": {"initiation": True, "detection": True, "failure": False},
                "condition": {
                    "condition_1": {"lower": 0, "upper": 50},
                    "condition_2": {"lower": None, "upper": 90},
                },
            }
        )

        # Act
        actual = task.sim_timeline(
            t_end=t_end, timeline=timeline, indicators=indicators
        )

        # Assert
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
