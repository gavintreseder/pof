"""
Tests for the FailureMode class
Author: Gavin Treseder
"""

import copy
import unittest
from unittest.mock import Mock, MagicMock, patch

import numpy as np

import fixtures
import testconfig
from test_load import TestPofBase
from pof.failure_mode import FailureMode
from pof.task import Task, ScheduledTask, ConditionTask, Inspection
import pof.demo as demo


def side_effect_trigger_task(**kwargs):
    t_start = kwargs.get("t_start")
    t_end = kwargs.get("t_end")

    return np.full(t_end - t_start + 1, 0)


class TestFailureMode(TestPofBase, unittest.TestCase):
    def setUp(self):

        super().setUp()

        # TestIntantiate
        self._class = FailureMode

        # TestFromDict
        self._data_valid = [{"name": "TestFailureMode"}]
        self._data_invalid_values = [{"pf_curve": "invalid_value"}]
        self._data_invalid_types = [{"invalid_type": "invalid_type"}]
        self._data_complete = [
            fixtures.complete["failure_mode_0"],
            fixtures.complete["failure_mode_1"],
        ]

    # ************ Test init_timeline ***********************

    def test_init_timeline(self):
        # TODO full coverage

        params = [
            ("step", demo.failure_mode_data["random"]),
            ("linear", demo.failure_mode_data["slow_aging"]),
        ]

        for pf_curve, test_data in params:
            # Arrange
            t_start = 0
            t_end = 200
            fm = FailureMode.load(demo.failure_mode_data["random"])

            # Act
            fm.init_timeline(t_start=0, t_end=200)

            # Check times match
            self.assertEqual(fm.timeline["time"][0], t_start, "t=0 != t_start")
            self.assertEqual(
                fm.timeline["time"][-1],
                t_end,
                "Last time in timeline does not equal t_end",
            )

            # Check states match
            self.assertEqual(
                fm.timeline["initiation"][0],
                fm.is_initiated(),
                "First initiation in timeline does not equal current initiation",
            )
            self.assertEqual(
                fm.timeline["detection"][0],
                fm.is_detected(),
                "First detection in timeline does not equal current detection",
            )
            self.assertEqual(
                fm.timeline["failure"][0],
                fm.is_failed(),
                "First Failure in timeline does not equal current failure",
            )

            # Check conditions match
            # TODO move conditions to indicators first

            # Check tasks match
            # TODO rewrite time function in tasks first

    # ------------ Test update_timeline --------------------

    def test_update_timeline(self):

        # Arrange
        fm = FailureMode.demo()

        fm.update_timeline(t_start=10, updates=dict(initiation=False))

        fm.update_timeline(t_start=5, updates=dict(initiation=False))

    # -------------Test sim_timleine ----------------------

    def test_sim_timeline_task_on_condition_replacement(self):
        """
        Check an on condition replacement task is triggered when the conditions are met
        """

        # Arrange so replacement should occur immediately
        fm = FailureMode.demo()
        fm.indicators["slow_degrading"].set_condition(10)
        fm.indicators["fast_degrading"].set_condition(10)
        fm.set_states(dict(initiation=True, detection=True))

        # Act
        fm.sim_timeline(200)

        # Assert task is triggered and reset occurs
        self.assertEqual(fm.timeline["on_condition_replacement"][0], 0, "no triggered")
        for state, value in (
            fm.tasks["on_condition_replacement"].impacts["state"].items()
        ):
            self.assertEqual(fm.timeline[state][1], value, "impact not completed")

    def test_sim_timeline_task_on_failure_replacement(self):
        # Arrange
        fm = FailureMode(
            untreated=fixtures.distribution_data["slow_aging"],
            conditions={
                "slow_degrading": fixtures.condition_data["slow_degrading"],
                "fast_degrading": fixtures.condition_data["fast_degrading"],
            },
            tasks={"replacement": fixtures.replacement_data["on_failure"]},
        )
        fm.set_states({"initiation": True})
        fm.indicators["slow_degrading"].set_condition(10)
        fm.indicators["fast_degrading"].set_condition(10)

        # Act
        fm.sim_timeline(200)

        # Assert task is triggered and reset occurs
        self.assertEqual(
            fm.timeline["on_failure_replacement"][0],
            -1,
            "task should not trigger at the t=0",
        )
        self.assertEqual(
            fm.timeline["on_failure_replacement"][1], 0, "task should trigger at t=1"
        )

        for state, value in fm.tasks["on_failure_replacement"].impacts["state"].items():
            self.assertEqual(fm.timeline[state][2], value, "impact not completed")

    def test_sim_timeline_remain_failed(self):
        """
        Check a timeline is impacted by the 'remain_failed' flag
        """

        params = [(True, 1, 0, False), (False, 2001, 0, True)]
        t_end = 2000

        for remain_failed, time_sim, time_failed, more_tasks in params:

            # Arrange so replacement should occur immediately
            fm = FailureMode.demo()
            fm.indicators["slow_degrading"].set_condition(10)
            fm.indicators["fast_degrading"].set_condition(10)
            fm.set_states(dict(initiation=True, detection=True))

            # Trigger tasks
            fm.dists["init"].sample = Mock(return_value=0)
            for task in fm.tasks.values():
                task.sim_timeline = Mock(side_effect=side_effect_trigger_task)

            # Act
            with patch.dict("pof.failure_mode.cf", {"remain_failed": remain_failed}):
                fm.sim_timeline(2000)

                # Assert
                self.assertEqual(len(fm.timeline["time"]), time_sim)
                self.assertEqual(
                    fm.timeline["on_condition_replacement"][0],
                    time_failed,
                    f"task should trigger at t=1",
                )
                for task_name in fm.tasks:
                    self.assertEqual(
                        any(fm.timeline[task_name][1:] + 1),
                        more_tasks,
                        "task should not be triggered again",
                    )

    # ************ Test sim_timeline ***********************

    def test_sim_timeline_condition_step(self):  # TODO full coverage
        t_start = 0
        t_end = 200
        fm = FailureMode.load(demo.failure_mode_data["random"])

        initiation_start = fm.is_initiated()
        detection_start = fm.is_detected()
        failure_start = fm.is_failed()

        fm.sim_timeline(t_start=t_start, t_end=t_end)

        # Check times are ok
        self.assertEqual(
            fm.timeline["time"][0], t_start, "First time does not equal t_start"
        )
        self.assertEqual(
            fm.timeline["time"][-1], t_end, "Last time in timeline does not equal t_end"
        )

        # Check states are ok
        self.assertEqual(
            fm.timeline["initiation"][0],
            initiation_start,
            "First initiation in timeline does not equal current initiation",
        )
        self.assertEqual(
            fm.timeline["initiation"][-1],
            fm.is_initiated(),
            "Last initiation in timeline does not equal current initiation",
        )
        self.assertEqual(
            fm.timeline["detection"][0],
            detection_start,
            "First detection in timeline does not equal current detection",
        )
        self.assertEqual(
            fm.timeline["detection"][-1],
            fm.is_detected(),
            "Last detection in timeline does not equal current detection",
        )
        self.assertEqual(
            fm.timeline["failure"][0],
            failure_start,
            "First Failure in timeline does not equal current failure",
        )
        self.assertEqual(
            fm.timeline["failure"][-1],
            fm.is_failed(),
            "Last Failure in timeline does not equal current failure",
        )

        # Check conditions match
        # TODO move conditions to indicators first

        # Check tasks match
        # TODO rewrite time function in tasks first

    def test_demo(self):
        fm = FailureMode.demo()
        self.assertIsNotNone(fm)

    # ------------ Test mc_timeline ------------------

    def test_mc_timeline(self):

        # Arrange
        fm = FailureMode.demo()

        # Act
        fm.mc_timeline(t_end=20, n_iterations=10)

    def test_mc_timeline_risk_is_accurate(self):
        fm = FailureMode.demo()

        fm.mc_timeline(200)

    # ************ Test get_dash_ids *****************

    def test_get_dash_id(self):

        fm = FailureMode.demo()

        dash_ids = fm.get_dash_ids()

    # ************ Test update methods *****************

    def test_update_on_property_method(self):

        # Arrange
        fm1 = FailureMode.demo()
        fm2 = FailureMode.demo()
        test_data = {
            "untreated": {"name": "untreated", "alpha": 20, "beta": 10, "gamma": 5}
        }

        # Act
        fm1.untreated = test_data["untreated"]
        fm2.update(test_data)

        # Assert
        self.assertEquals(fm1, fm2)

    # ************ Test link indicators ***************

    # TODO change to use set methods

    def test_link_indicators_if_present(self):

        fm1 = FailureMode.from_dict(demo.failure_mode_data["early_life"])
        fm2 = FailureMode.from_dict(demo.failure_mode_data["random"])
        fm3 = FailureMode.from_dict(demo.failure_mode_data["slow_aging"])
        fm4 = FailureMode.from_dict(demo.failure_mode_data["fast_aging"])

        fm1.link_indicator(fm2.conditions["instant"])
        fm1.conditions["instant"].pf_interval = -100
        fm2.conditions["instant"].pf_interval = -1000

        fm3.link_indicator(fm4.conditions["slow_degrading"])
        fm3.sim_timeline(200)

        self.assertEqual(
            fm1.conditions,
            fm2.conditions,
            msg="Indicators should be the same after values are assigned",
        )
        self.assertEqual(
            fm3.conditions["slow_degrading"],
            fm4.conditions["slow_degrading"],
            msg="Indicators should be the same after methods are executed",
        )
        self.assertNotEqual(
            fm3.conditions["fast_degrading"],
            fm4.conditions["fast_degrading"],
            msg="Only named indicators should be linked",
        )

    def test_link_indicators_if_not_present(self):

        fm1 = FailureMode.from_dict(demo.failure_mode_data["early_life"])
        fm2 = FailureMode.from_dict(demo.failure_mode_data["random"])
        fm3 = FailureMode.from_dict(demo.failure_mode_data["slow_aging"])
        fm4 = FailureMode.from_dict(demo.failure_mode_data["fast_aging"])

        fm1.link_indicator(fm2.conditions["instant"])
        fm3.link_indicator(fm2.conditions["instant"])
        fm4.link_indicator(fm2.conditions["instant"])
        fm1.conditions["instant"].pf_interval = -100
        fm3.sim_timeline(100)

        with self.assertRaises(
            KeyError,
            msg="Indicator should not be able to link if there isn't an indicator by that name",
        ):
            fm3.conditions["instant"].pf_interval = 200

        self.assertEqual(fm1.conditions, fm2.conditions)
        self.assertNotEqual(fm3.conditions, fm4.conditions)

    # ************ Test reset methods *****************

    # Change all condition, state and task count. Check values change or don't change for each of them

    # def test_update(self):

    #     test_data_1_fix = fixtures.failure_mode_data["early_life"]
    #     test_data_2_fix = fixtures.failure_mode_data["random"]

    #     fm1 = FailureMode.from_dict(test_data_1_fix)
    #     fm2 = FailureMode.from_dict(test_data_2_fix)

    #     fm1.update_from_dict(test_data_2_fix)

    #     self.assertEqual(fm1, fm2)

    def test_set_task_Task(self):

        fm = FailureMode.from_dict(fixtures.failure_mode_data["random"])

        test_data = Task.from_dict(fixtures.inspection_data["instant"])

        fm.set_obj("tasks", Task, test_data)

        self.assertIsInstance(fm.tasks["inspection"], Task)
        self.assertEqual(
            fm.tasks["inspection"].cost,
            test_data.cost,
        )

    def test_set_task_dict_Task(self):

        fm = FailureMode.from_dict(fixtures.failure_mode_data["random"])

        test_data = dict(inspection=Task.from_dict(fixtures.inspection_data["instant"]))

        fm.set_obj("tasks", Task, test_data)

        self.assertIsInstance(fm.tasks["inspection"], Task)
        self.assertEqual(
            fm.tasks["inspection"].cost,
            test_data["inspection"].cost,
        )

    def test_set_task_dict_update(self):

        fm = FailureMode.from_dict(fixtures.failure_mode_data["random"])

        test_data = dict(inspection=fixtures.inspection_data["instant"])

        fm.set_obj("tasks", Task, test_data)

        self.assertIsInstance(fm.tasks["inspection"], Task)
        self.assertEqual(
            fm.tasks["inspection"].cost,
            test_data["inspection"]["cost"],
        )

    def test_set_task_dict(self):

        fm = FailureMode.from_dict(fixtures.failure_mode_data["random"])

        test_data = fixtures.inspection_data["instant"]

        fm.set_obj("tasks", Task, test_data)

        self.assertIsInstance(fm.tasks["inspection"], Task)
        self.assertEqual(
            fm.tasks["inspection"].cost,
            test_data["cost"],
        )

    # ------------------ Test Expected Risk ------------------------

    def test_expected_risk(self):

        fm = FailureMode.demo()

        fm._timeline["failure"] = np.full(0, 200)

        er = fm._expected_risk()

        np.testing.assert_array_equal(er["time"], [])

    # ------------ Integration Tests ---------------

    def test_init_dist_is_updated_on_creation(self):

        # Arrange / Act
        fm = FailureMode.from_dict(
            {"pf_interval": 10, "untreated": {"alpha": 100, "beta": 10, "gamma": 10}}
        )

        # Assert
        self.assertNotEqual(fm.untreated, fm.dists["init"])

    def test_init_dist_is_updated_with_untreated(self):

        # Arrange
        fm = FailureMode.from_dict(
            {"pf_interval": 10, "untreated": {"alpha": 100, "beta": 10, "gamma": 10}}
        )
        untreated = copy.copy(fm.dists.get("untreated", None))
        init = copy.copy(fm.dists.get("init", None))

        # Act
        fm.untreated = {"alpha": 50, "beta": 10, "gamma": 5}

        # Assert
        self.assertNotEqual(fm.dists["untreated"], untreated)
        self.assertNotEqual(fm.dists["init"], init)

    def test_init_dist_is_updated_with_pf_interval(self):

        # Arrange
        fm = FailureMode.from_dict(
            {"pf_interval": 10, "untreated": {"alpha": 100, "beta": 10, "gamma": 10}}
        )
        untreated = copy.copy(fm.dists.get("untreated", None))
        init = copy.copy(fm.dists.get("init", None))

        # Act
        fm.pf_interval = 5

        # Assert
        self.assertEqual(fm.dists["untreated"], untreated)
        self.assertNotEqual(fm.dists["init"], init)

    def test_init_dist_is_updated_with_update(self):
        """ Check that init dist updates when the update method is called on any of its inputs"""
        param_update = [
            {"pf_interval": 5},
            {"untreated": {"alpha": 80}},
        ]

        for update in param_update:

            # Arrange
            fm = FailureMode.from_dict(
                {
                    "pf_interval": 10,
                    "untreated": {"alpha": 100, "beta": 10, "gamma": 10},
                }
            )
            init = copy.copy(fm.dists.get("init", None))

            # Act
            fm.update(update)

            # Assert
            self.assertNotEqual(fm.dists["init"], init)


if __name__ == "__main__":
    unittest.main()
