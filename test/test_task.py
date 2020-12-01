"""
    Filename: test_task.py
    Description: Contains the code for testing the Task class
    Author:
        Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
        Illyse Schram  | ischram@kpmg.com.au | illyse.schram@essentialenergy.com.au
"""

import unittest
from unittest.mock import Mock

import numpy as np

import fixtures
import testconfig  # pylint: disable=unused-import
from test_load import TestPofBase
from pof.task import Task, ScheduledTask, ConditionTask, Inspection


class TestTaskCommon(TestPofBase):
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
    def setUp(self):
        super().setUp()

        self._class = Task
        self._data_valid = [dict(name="TaskTest", task_type="Task")]
        self._data_invalid_types = [
            dict(invalid_input="invalid_input", task_type="Task")
        ]
        self._data_complete = [
            fixtures.complete["task_0"],
            fixtures.complete["task_1"],
        ]


class TestScheduledTask(TestTaskCommon, unittest.TestCase):
    """
    Tests the functionality of the ScheduledTask class including all common tests written in TestTaskCommon
    """

    def setUp(self):
        super().setUp()

        self._class = ScheduledTask
        self._data_valid = [{"name": "ScheduledTaskTest", "task_type": "ScheduledTask"}]
        self._data_invalid_types = [
            dict(invalid_input="invalid_input", task_type="ScheduledTask")
        ]
        self._data_complete = [
            fixtures.complete["scheduled_task_0"],
            fixtures.complete["scheduled_task_1"],
        ]

    def test_sim_timeline(self):
        """Check the a scheduled task returns the correct time"""
        # Task params
        param_t_delay = [0, 1, 5]
        param_t_interval = [1, 3, 5]  # TODO should it work with 0?

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
                        delay = np.linspace(t_delay, 0, t_delay + 1)

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
        self._data_valid = [dict(name="TestInspection", task_type="ConditionTask")]
        self._data_invalid_types = [
            {"invalid_input": "invalid_input", "task_type": "ConditionTask"}
        ]
        self._data_complete = [
            fixtures.complete["condition_task_0"],
            fixtures.complete["condition_task_1"],
        ]

    # **************** test_load ***********************


class TestInspection(TestTaskCommon, unittest.TestCase):
    def setUp(self):
        super().setUp()

        # TestTaskCommon Setup
        self._class = Inspection
        self._data_valid = [dict(name="TestInspection", task_type="Inspection")]
        self._data_invalid_types = [
            dict(invalid_input="invalid_input", task_type="Inspection")
        ]
        self._data_complete = [
            fixtures.complete["inspection_0"],
            fixtures.complete["inspection_1"],
        ]

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
