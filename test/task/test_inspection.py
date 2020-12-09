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

import fixtures
import testconfig  # pylint: disable=unused-import
from task.test_task import TestTaskCommon
from pof.task import Inspection


class TestInspection(TestTaskCommon, unittest.TestCase):
    """
    Tests the functionality of the Inspection class including all common tests written in TestTaskCommon
    """

    def setUp(self):
        super().setUp()

        # TestTaskCommon Setup
        self._class = Inspection
        self._data_valid = [dict(name="TestInspection", task_type="Inspection")]
        self._data_invalid_types = [
            dict(invalid_input="invalid_input", task_type="Inspection")
        ]
        self._data_complete = copy.deepcopy(fixtures.complete["inspection"])

    def test_effectiveness(self):
        """ Check the probability of effectiveness is calculated correctly"""

        # p_effective, pf_interval, t_delay, t_interval, expected
        param_list = [
            # (0, 0, 0, 0, 0),  # All Zero
            (0, 4, 0, 5, 0),  # Not effective
            (1, 4, 0, 5, 0.8),  # Not enough inspections
            (1, 5, 0, 5, 1),  # Enough inspections
            (1, 6, 0, 5, 1),  # Too many inspections
            (0.5, 10, 0, 5, 0.75),  # Prob failure
            (0.5, 10, 10, 5, 0.675),
        ]

        failure_dist = Mock()
        failure_dist.cdf = Mock(return_value=[0.1])

        for p_effective, pf_interval, t_delay, t_interval, expected in param_list:

            # Arrange
            inspection = Inspection(
                p_effective=p_effective, t_interval=t_interval, t_delay=t_delay
            )

            # Act
            actual = inspection.effectiveness(pf_interval, failure_dist=failure_dist)

            # Assert
            self.assertEqual(expected, actual)

    # **************** test_update ***********************


if __name__ == "__main__":
    unittest.main()
