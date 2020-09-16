import unittest
import unittest.mock
from unittest.mock import patch

import utils

from pof.failure_mode import FailureMode
from pof.config import FailureModeConfig as cf
import pof.demo as demo


class TestFailureMode(unittest.TestCase):
    def test_imports_correctly(self):
        self.assertTrue(True)

    def test_instantiate(self):
        try:
            fm = FailureMode()
            self.assertIsNotNone(fm)
        except ValueError:
            self.fail("ValueError returned")
        except:
            self.fail("Unknown error")

    def test_from_dict(self):
        try:
            fm = FailureMode.from_dict(demo.failure_mode_data["early_life"])
            self.assertIsNotNone(fm)
        except ValueError:
            self.fail("ValueError returned")
        except:
            self.fail("Unknown error")

    def test_from_dict_value_error_exits(self):

        false_data = dict(pf_curve="invalid_value")

        with self.assertRaises(ValueError):
            fm = FailureMode.from_dict(false_data)
            self.assertIsNotNone(fm)

    def test_instantiate_with_data(self):
        try:
            fm = FailureMode(name="random", untreated=dict(alpha=500, beta=1, gamma=0))
            self.assertIsNotNone(fm)
        except ValueError:
            self.fail("ValueError returned")
        except:
            self.fail("Unknown error")

    # ************ Test init_timeline ***********************

    def test_init_timeline_condition_step(self):  # TODO full coverage
        t_start = 0
        t_end = 200
        fm = FailureMode().load(demo.failure_mode_data["random"])

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
        fm = FailureMode(demo.failure_mode_data["slow_aging"])

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

    # ************ Test load ***********************

    def test_load_data_demo_data(self):
        try:
            fm = FailureMode.load(demo.failure_mode_data["slow_aging"])
            self.assertIsNotNone(fm)
        except ValueError:
            self.fail("ValueError returned")
        except:
            self.fail("Unknown error")

    def test_load_demo_no_data(self):
        try:
            fm = FailureMode.load()
            self.assertIsNotNone(fm)
        except ValueError:
            self.fail("ValueError returned")
        except:
            self.fail("Unknown error")

    def test_set_demo_some_data(self):
        fm = FailureMode().set_demo()
        self.assertIsNotNone(fm)

    # ************ Test Dash ID Value ***********************

    def test_get_dash_id_value(self):

        fm = FailureMode(alpha=50, beta=1.5, gamma=10).set_demo()

        dash_ids = fm.get_dash_ids()

        # TODO load data

    # ************ Test get_dash_ids *****************

    def test_get_dash_ids(self):

        fm = FailureMode()

    # ************ Test update methods *****************

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


if __name__ == "__main__":
    unittest.main()