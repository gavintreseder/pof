"""
#TODO add docstring
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import logging
import sys
import numpy as np

import utils
from pof.failure_mode import FailureMode
from pof.task import Task, ScheduledTask, ConditionTask, Inspection
import pof.demo as demo
import fixtures


class TestFailureMode(unittest.TestCase):
    def setUp(self):
        self.blank_config = Mock()
        self.blank_config.get.return_value = None
        self.blank_config.getboolean.return_value = None
        self.blank_config.getint.return_value = None

    def test_class_imports_correctly(self):
        self.assertIsNotNone(FailureMode)

    def test_class_instantiate_no_data(self):
        failure_mode = FailureMode()
        self.assertIsNotNone(failure_mode)

    def test_class_instantiate_with_valid_data(self):
        failure_mode = FailureMode(
            name="random",
            untreated=dict(name="slow_degrading", alpha=500, beta=1, gamma=0),
        )
        self.assertIsNotNone(failure_mode)

    def test_class_instantiate_with_invalid_data(self):

        invalid_data_type = dict(invalid_input="invalid_value")
        invalid_data_value = dict(pf_curve="incorrect_value")

        with self.assertRaises(TypeError):
            FailureMode.from_dict(invalid_data_type)

        # TODO patch this method of on error use default
        with patch("pof.load.cf", self.blank_config):
            with self.assertRaises(ValueError):
                FailureMode.from_dict(invalid_data_value)

    def test_from_dict_no_data(self):
        with self.assertRaises(TypeError):
            FailureMode.from_dict()

    def test_from_dict_with_valid_data(self):
        fm = FailureMode.from_dict(fixtures.failure_mode_data["early_life"])
        self.assertIsNotNone(fm)

    def test_from_dict_with_invalid_data_config_default(self):
        # TODO Mock cf.get_boolean('on_error_default')
        invalid_data = dict(pf_curve="invalid_value")

        with patch("pof.failure_mode.cf", Mock()):
            with self.assertRaises(ValueError):
                FailureMode.from_dict(invalid_data)

    def test_from_dict_with_invalid_data_config_none(self):

        invalid_data = dict(pf_curve="invalid_value")
        with patch("pof.failure_mode.cf", self.blank_config):
            with patch("pof.load.cf", self.blank_config):
                with self.assertRaises(ValueError):
                    FailureMode.from_dict(invalid_data)

    # ************ Test init_timeline ***********************

    def test_init_timeline_condition_step(self):  # TODO full coverage
        t_start = 0
        t_end = 200
        fm = FailureMode.load(demo.failure_mode_data["random"])

        fm.init_timeline(t_start=0, t_end=200)

        # Check times match
        self.assertEqual(
            fm.timeline["time"][0], t_start, "First time does not equal t_start"
        )
        self.assertEqual(
            fm.timeline["time"][-1], t_end, "Last time in timeline does not equal t_end"
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

    def test_init_timeline_condition_linear(self):  # TODO full coverage
        t_start = 0
        t_end = 200
        fm = FailureMode.load(demo.failure_mode_data["slow_aging"])
        fm2 = FailureMode.load(demo.failure_mode_data["slow_aging"])

        fm.init_timeline(t_start=0, t_end=200)

        # Check times match
        self.assertEqual(
            fm.timeline["time"][0],
            t_start,
            "First time in timeline should equal _start",
        )
        self.assertEqual(
            fm.timeline["time"][-1], t_end, "Last time in timeline should equal t_end"
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
        # TODO move conditions to indicators first copy from previous test

        # Check tasks match
        # TODO rewrite time function in tasks first copy from previous test

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

    def test_sim_timeline_on_condition_task_triggered(self):
        # Arrange
        fm = FailureMode(
            untreated=fixtures.distribution_data["slow_aging"],
            conditions={
                "slow_degrading": fixtures.condition_data["slow_degrading"],
                "fast_degrading": fixtures.condition_data["fast_degrading"],
            },
            tasks={"insepction": fixtures.replacement_data["on_condition"]},
        )
        fm.set_states({"initiation": True})
        fm.indicators["slow_degrading"].set_condition(10)
        fm.indicators["fast_degrading"].set_condition(10)

        # Act
        fm.sim_timeline(200)

        # Assert
        fm

        # self.assertEqual()

    # ************ Test load ***********************

    def test_load(self):
        fm = FailureMode.load()
        self.assertIsNotNone(fm)

    def test_load_no_data_no_config(self):
        with patch("pof.failure_mode.cf", self.blank_config):
            with self.assertRaises(
                ValueError,
                msg="Error expected with no input",
            ):
                FailureMode.load()

    def test_load_data_demo_data(self):
        try:
            fm = FailureMode.load(demo.failure_mode_data["slow_aging"])
            self.assertIsNotNone(fm)
        except ValueError:
            self.fail("ValueError returned")
        except:
            self.fail("Unknown error")

    def test_set_demo_some_data(self):
        fm = FailureMode.demo()
        self.assertIsNotNone(fm)

    # ************ Test get_dash_ids *****************

    def test_get_dash_id(self):

        fm = FailureMode.demo()

        dash_ids = fm.get_dash_ids()

    # ************ Test update methods *****************

    def test_update(self):
        # TODO This will be causing errors because new values are being created
        test_data_1 = fixtures.failure_mode_data["early_life"]
        test_data_2 = fixtures.failure_mode_data["slow_aging"]

        fm1 = FailureMode.from_dict(test_data_1)
        fm2 = FailureMode.from_dict(test_data_2)

        # update var by var
        for var, val in test_data_2.items():
            d = {}
            d[var] = val

            fm1.update(d)

        self.assertEqual(fm1, fm2)

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

        fm._set_container_attr("tasks", Task, test_data)

        self.assertIsInstance(fm.tasks["inspection"], Task)
        self.assertEqual(
            fm.tasks["inspection"].cost,
            test_data.cost,
        )

    def test_set_task_dict_Task(self):

        fm = FailureMode.from_dict(fixtures.failure_mode_data["random"])

        test_data = dict(inspection=Task.from_dict(fixtures.inspection_data["instant"]))

        fm._set_container_attr("tasks", Task, test_data)

        self.assertIsInstance(fm.tasks["inspection"], Task)
        self.assertEqual(
            fm.tasks["inspection"].cost,
            test_data["inspection"].cost,
        )

    def test_set_task_dict_update(self):

        fm = FailureMode.from_dict(fixtures.failure_mode_data["random"])

        test_data = dict(inspection=fixtures.inspection_data["instant"])

        fm._set_container_attr("tasks", Task, test_data)

        self.assertIsInstance(fm.tasks["inspection"], Task)
        self.assertEqual(
            fm.tasks["inspection"].cost,
            test_data["inspection"]["cost"],
        )

    def test_set_task_dict(self):

        fm = FailureMode.from_dict(fixtures.failure_mode_data["random"])

        test_data = dict(fixtures.inspection_data["instant"])

        fm._set_container_attr("tasks", Task, test_data)

        self.assertIsInstance(fm.tasks["inspection"], Task)
        self.assertEqual(
            fm.tasks["inspection"].cost,
            test_data["cost"],
        )

    # ------------------ Test Expected Risk ------------------------

    def test_mc_timeline_risk_is_accurate(self):
        fm = FailureMode.demo()

        fm.mc_timeline(200)

    def test_expected_risk(self):

        fm = FailureMode.demo()

        fm._timeline["failure"] = np.full(False, 200)

        er = fm._expected_risk()

        np.testing.assert_array_equal(er["time"], [])


if __name__ == "__main__":
    unittest.main()
