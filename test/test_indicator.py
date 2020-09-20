import unittest
from unittest.mock import Mock
import numpy as np
import copy

import utils

from pof.indicator import ConditionIndicator
import pof.demo as demo

import fixtures


cid = dict(
    name="test",
    perfect=100,
    failed=0,
    pf_curve="linear",
    pf_interval=100,
    pf_std=None,
)


class TestConditionIndicator(unittest.TestCase):
    def test_class_imports_correctly(self):
        self.assertTrue(True)

    def test_instantiate(self):
        try:
            cond = ConditionIndicator()
            self.assertIsNotNone(cond)
        except ValueError:
            self.fail("ValueError returned")
        except:
            self.fail("Unknown error")

    def test_instantiate_with_data(self):
        try:
            cond = ConditionIndicator(name="test_name")
            self.assertIsNotNone(cond)
        except ValueError:
            self.fail("ValueError returned")
        except:
            self.fail("Unknown error")

    def test_from_dict(self):
        try:
            cond = ConditionIndicator.from_dict()
            self.assertIsNotNone(cond)
        except ValueError:
            self.fail("ValueError returned")
        except:
            self.fail("Unknown error")

    def test_from_dict_value_error_exits(self):

        false_data = dict(pf_curve="incorrect_value")

        with self.assertRaises(ValueError) as cm:
            cond = ConditionIndicator.from_dict(false_data)

    def test_from_dict_with_data(self):
        try:
            cond = ConditionIndicator.from_dict(demo.condition_data["instant"])
            self.assertIsNotNone(cond)
        except ValueError:
            self.fail("ValueError returned")
        except:
            self.fail("Unknown error")

    # Next test

    def test_sim_timeline_two_causes(self):

        cond = ConditionIndicator()

        cond.sim_timeline(t_stop=200, name="cause_1")
        cond.sim_timeline(t_stop=200, name="cause_2")

        timeline_1 = cond.get_timeline(name="cause_1")
        timeline_2 = cond.get_timeline(name="cause_2")

    # ********************* Test sim_timeline *********************

    def test_sim_timeline(self):
        cond = ConditionIndicator()
        cond.sim_timeline(t_stop=100)
        self.assertIsNotNone(cond)

    def test_sim_timeline_no_data(self):
        cond = ConditionIndicator()
        cond.sim_timeline()
        self.assertIsNotNone(cond)

    def test_sim_timeline_no_data_passed(self):
        cond = ConditionIndicator.from_dict(cid)
        cond.sim_timeline()

    def test_sim_timeline_order_none_then_data(self):

        mock = Mock()
        mock(100)
        mock(t_stop=100, pf_interval=10, name="cause_1")
        mock(t_stop=100, pf_interval=20, name="cause_1")

        expected = {
            0: np.linspace(100, 0, 101),
            1: np.concatenate([np.linspace(100, 0, 11), np.full(90, 0)]),
            2: np.concatenate([np.linspace(100, 0, 21), np.full(80, 0)]),
        }

        cond = ConditionIndicator.from_dict(cid)

        for k, v in expected.items():
            # Unpack the test data
            args, kwargs = mock.call_args_list[k]
            name = mock.call_args_list[k].kwargs.get("name", None)
            cond.sim_timeline(*args, **kwargs)  # TODO check why timeline isn't updating
            np.testing.assert_array_equal(
                cond.get_timeline(name),
                v,
                err_msg="Failed with %s" % (str(mock.call_args_list[k])),
            )

    def test_sim_timeline_order_data_then_none(self):
        """
        Checks that a the same name gets overwritten and that on get_timline returns
        """
        mock = Mock()
        mock(t_stop=100, pf_interval=10, name="cause_1")
        mock(t_stop=100, pf_interval=400, name="cause_1")
        mock(t_stop=100, pf_interval=400, name="cause_2")
        mock(t_stop=100, pf_interval=200)

        expected = {
            0: np.concatenate([np.linspace(100, 0, 11), np.full(90, 0)]),
            1: np.linspace(100, 75, 101),
            2: np.linspace(100, 75, 101),
            3: np.linspace(100, 50, 101),
        }

        cond = ConditionIndicator.from_dict(cid)

        for key, val in expected.items():
            # Unpack the test data
            args, kwargs = mock.call_args_list[key]
            name = mock.call_args_list[key].kwargs.get("name", None)
            cond.sim_timeline(*args, **kwargs)  # TODO check why timeline isn't updating
            self.assertIsNone(
                np.testing.assert_array_equal(
                    cond.get_timeline(name),
                    val,
                    err_msg="Failed with %s" % (str(mock.call_args_list[key])),
                )
            )

    # *********** Test the condition limits ******************

    def test_condition_starts_zero(self):
        cond = ConditionIndicator(
            perfect=0, failed=100, pf_curve="linear", pf_interval=10
        )
        self.assertEqual(cond.get_condition(), 0)

    def test_condition_starts_zero_does_not_breach_limit(self):
        cond = ConditionIndicator(
            perfect=0, failed=100, pf_curve="linear", pf_interval=10
        )
        cond.sim(100)
        self.assertEqual(cond.get_condition(), 100)

    def test_condition_starts_positive(self):
        cond = ConditionIndicator(
            perfect=100, failed=0, pf_curve="linear", pf_interval=10
        )
        self.assertEqual(cond.get_condition(), 100)

    def test_condition_starts_positive_does_not_breach_limit(self):
        cond = ConditionIndicator(
            perfect=100, failed=0, pf_curve="linear", pf_interval=10
        )
        cond.sim_timeline(100)

        self.assertEqual(min(cond.get_timeline()), 0)

    # ********** Test sim_timeline **********

    # early start

    def test_sim_timeline_early_start_early_stop(self):
        expected = np.concatenate((np.full(10, 100), np.linspace(100, 90, 11)))
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )

        cp = cond.sim_timeline(t_start=-10, t_stop=10)

        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_early_start_late_stop(self):
        expected = np.concatenate(
            (np.full(10, 100), np.linspace(100, 50, 51), np.full(50, 50))
        )
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )

        cp = cond.sim_timeline(t_start=-10, t_stop=100)

        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_early_start_no_stop(self):
        expected = np.concatenate((np.full(10, 100), np.linspace(100, 50, 51)))
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )

        cp = cond.sim_timeline(t_start=-10)

        np.testing.assert_array_equal(cp, expected)

    # existing condition

    def test_sim_timeline_existing_condition_early_start_early_stop(self):
        expected = np.concatenate((np.full(10, 90), np.linspace(90, 70, 21)))
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )

        cond.set_condition(90)
        cp = cond.sim_timeline(t_start=-10, t_stop=20)

        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_existing_condition_early_start_late_stop(self):
        expected = np.concatenate(
            (np.full(10, 90), np.linspace(90, 50, 41), np.full(60, 50))
        )
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )

        cond.set_condition(90)
        cp = cond.sim_timeline(t_start=-10, t_stop=100)

        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_existing_condition_early_start_no_stop(self):
        expected = np.concatenate((np.full(10, 90), np.linspace(90, 50, 41)))
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )

        cond.set_condition(90)
        cp = cond.sim_timeline(t_start=-10)

        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_existing_condition_v_early_start_v_late_stop(self):
        expected = np.concatenate(
            (np.full(10, 90), np.linspace(90, 50, 41), np.full(60, 50))
        )
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )

        cond.set_condition(90)
        cp = cond.sim_timeline(t_start=-10, t_stop=100)
        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_existing_condition_v_late_start_v_late_stop(self):
        expected = np.full(41, 50)
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )

        cond.set_condition(50)
        cp = cond.sim_timeline(t_start=60, t_stop=100)

        np.testing.assert_array_equal(cp, expected)

    # late start

    def test_sim_timeline_late_start_early_stop(self):
        expected = np.linspace(95, 90, 6)
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )

        cp = cond.sim_timeline(t_start=5, t_stop=10)

        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_late_start_late_stop(self):
        expected = np.concatenate((np.linspace(95, 50, 46), np.full(50, 50)))
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )

        cp = cond.sim_timeline(t_start=5, t_stop=100)

        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_late_start_no_stop(self):
        expected = np.linspace(95, 50, 46)
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )

        cp = cond.sim_timeline(t_start=5)

        np.testing.assert_array_equal(cp, expected)

    # no start

    def test_sim_timeline_no_start_early_stop(self):
        expected = np.linspace(100, 90, 11)
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )

        cp = cond.sim_timeline(t_stop=10)

        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_no_start_late_stop(self):
        expected = np.concatenate((np.linspace(100, 50, 51), np.full(50, 50)))
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )

        cp = cond.sim_timeline(t_stop=100)

        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_no_start_no_stop(self):
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )
        expected = np.linspace(100, 50, 51)
        cp = cond.sim_timeline()
        np.testing.assert_array_equal(cp, expected)

    # very early start

    def test_sim_timeline_v_early_start_early_stop(self):
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )
        expected = np.concatenate((np.full(100, 100), np.linspace(100, 90, 11)))
        cp = cond.sim_timeline(t_start=-100, t_stop=10)
        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_v_early_start_late_stop(self):
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )
        expected = np.concatenate(
            (np.full(100, 100), np.linspace(100, 50, 51), np.full(50, 50))
        )
        cp = cond.sim_timeline(t_start=-100, t_stop=100)
        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_v_early_start_no_stop(self):
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )
        expected = np.concatenate((np.full(100, 100), np.linspace(100, 50, 51)))
        cp = cond.sim_timeline(t_start=-100)
        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_v_early_start_v_early_stop(self):
        expected = np.full(91, 100)
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )
        cp = cond.sim_timeline(t_start=-100, t_stop=-10)
        np.testing.assert_array_equal(cp, expected)

    # very late start

    def test_sim_timeline_v_late_start_early_stop(self):
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )
        expected = [90]
        cp = cond.sim_timeline(t_start=20, t_stop=10)
        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_v_late_start_late_stop(self):
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )
        expected = [50]
        cp = cond.sim_timeline(t_start=110, t_stop=100)
        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_v_late_start_no_stop(self):
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )
        expected = [50]
        cp = cond.sim_timeline(t_start=60)
        np.testing.assert_array_equal(cp, expected)

    # ********** Test set_condition **************** TODO add condition checks as well as time

    def test_set_condition_pf_decreasing_above_limit(self):
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )

        cond.set_condition(150)

        self.assertEqual(cond.get_condition(), 100)
        self.assertEqual(cond.get_accumulated(), 0)

    def test_set_condition_pf_decreasing_below_limit(self):
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )

        cond.set_condition(0)

        self.assertEqual(cond.get_accumulated(), 50)
        self.assertEqual(cond.get_condition(), 50)

    def test_set_condition_pf_decreasing_in_limit(self):
        cond = ConditionIndicator(
            perfect=100, failed=50, pf_interval=50, pf_curve="linear"
        )

        cond.set_condition(70)

        self.assertEqual(cond.get_condition(), 70)
        self.assertEqual(cond.get_accumulated(), 30)

    def test_set_condition_pf_increasing_above_limit(self):
        cond = ConditionIndicator(
            perfect=50, failed=100, pf_interval=50, pf_curve="linear"
        )

        cond.set_condition(150)

        self.assertEqual(cond.get_condition(), 100)
        self.assertEqual(cond.get_accumulated(), 50)

    def test_set_condition_pf_increasing_below_limit(self):
        cond = ConditionIndicator(
            perfect=50, failed=100, pf_interval=50, pf_curve="linear"
        )

        cond.set_condition(0)

        self.assertEqual(cond.get_condition(), 50)
        self.assertEqual(cond.get_accumulated(), 0)

    def test_set_condition_pf_increasing_in_limit(self):
        cond = ConditionIndicator(
            perfect=50, failed=100, pf_interval=50, pf_curve="linear"
        )

        cond.set_condition(70)

        self.assertEqual(cond.get_condition(), 70)
        self.assertEqual(cond.get_accumulated(), 20)

    # **************** Test limit_reached *****************

    """def test_is_failed_pf_decreasing_at_threshold (self):
        c = ConditionIndicator(100,0,'linear', [-10])
        c.threshold_failure = 50
        cond.set_condition(50)
        self.assertTrue(c.is_failed())

    def test_is_failed_pf_decreasing_exceeds_threshold (self):
        c = ConditionIndicator(100,0,'linear', [-10])
        c.threshold_failure = 50
        cond.set_condition(30)
        self.assertTrue(c.is_failed())

    def test_is_failed_pf_decreasing_within_threshold (self):
        c = ConditionIndicator(100,0,'linear', [-10])
        c.threshold_failure = 50
        cond.set_condition(70)
        self.assertFalse(c.is_failed())
    
    def test_is_failed_pf_increasing_at_threshold (self):
        c = ConditionIndicator(0,100,'linear', [10], decreasing = False)
        c.threshold_failure = 50
        cond.set_condition(50)
        self.assertTrue(c.is_failed())

    def test_is_failed_pf_increasing_exceeds_threshold (self):
        c = ConditionIndicator(0,100,'linear', [10], decreasing = False)
        c.threshold_failure = 50
        cond.set_condition(70)
        self.assertTrue(c.is_failed())

    def test_is_failed_pf_increasing_within_threshold (self):
        c = ConditionIndicator(0,100,'linear', [10], decreasing = False)
        c.threshold_failure = 50
        cond.set_condition(30)
        self.assertFalse(c.is_failed())"""

    # **************** Test the measuring functions

    def test_detectable_condition_starts_zero_current_above_detection_threshold_zero(
        self,
    ):
        cond = ConditionIndicator(
            perfect=0, failed=100, pf_curve="linear", pf_interval=10
        )
        cond.condition_detectable = 0
        cond.set_condition(30)
        self.assertTrue(cond.detectable())

    def test_detectable_condition_starts_zero_current_at_detection_threshold_zero(self):
        cond = ConditionIndicator(
            perfect=0, failed=100, pf_curve="linear", pf_interval=10
        )
        cond.condition_detectable = 0
        cond.set_condition(0)
        self.assertTrue(cond.detectable())

    def test_detectable_condition_starts_zero_current_above_detection_threshold_positive(
        self,
    ):
        cond = ConditionIndicator(
            perfect=0, failed=100, pf_curve="linear", pf_interval=10
        )
        cond.condition_detectable = 20
        cond.set_condition(30)
        self.assertTrue(cond.detectable())

    def test_detectable_condition_starts_zero_current_at_detection_threshold_positive(
        self,
    ):
        cond = ConditionIndicator(
            perfect=0, failed=100, pf_curve="linear", pf_interval=10
        )
        cond.condition_detectable = 20
        cond.set_condition(20)
        self.assertTrue(cond.detectable())

    def test_detectable_condition_starts_positive_current_below_detection_threshold_positive(
        self,
    ):
        cond = ConditionIndicator(
            perfect=100, failed=0, pf_curve="linear", pf_interval=10
        )
        cond.condition_detectable = 80
        cond.set_condition(cond.condition_perfect)
        self.assertFalse(cond.detectable())

    def test_detectable_condition_starts_positive_current_above_detection_threshold_zero(
        self,
    ):
        cond = ConditionIndicator(
            perfect=100, failed=0, pf_curve="linear", pf_interval=10
        )
        cond.condition_detectable = 80
        cond.set_condition(90)
        self.assertFalse(cond.detectable())

    def test_detectable_condition_starts_positive_current_at_detection_threshold_zero(
        self,
    ):
        cond = ConditionIndicator(
            perfect=100, failed=0, pf_curve="linear", pf_interval=10
        )
        cond.condition_detectable = 80
        cond.set_condition(0)
        self.assertTrue(cond.detectable())

    def test_detectable_condition_starts_positive_current_above_detection_threshold_positive(
        self,
    ):
        cond = ConditionIndicator(
            perfect=100, failed=0, pf_curve="linear", pf_interval=10
        )
        cond.condition_detectable = 80
        cond.set_condition(70)
        self.assertTrue(cond.detectable())

    def test_detectable_condition_starts_positive_current_at_detection_threshold_positive(
        self,
    ):
        cond = ConditionIndicator(
            perfect=100, failed=0, pf_curve="linear", pf_interval=10
        )
        cond.condition_detectable = 80
        cond.set_condition(80)
        self.assertTrue(cond.detectable())

    # **************** Test the reset functions ***********************

    def test_reset(self):
        cond = ConditionIndicator(
            perfect=100, failed=0, pf_curve="linear", pf_interval=10
        )
        cond.set_condition(50)
        cond.reset()
        self.assertEqual(cond.get_condition(), 100)

    # **************** Test the reset_any functions ***********************

    def test_reset_any_reduction_factor_all(self):
        cond = ConditionIndicator(
            perfect=100, failed=0, pf_curve="linear", pf_interval=10
        )

        cond.set_condition(50)
        cond.reset_any(target=1, method="reduction_factor", axis="condition")
        condition = cond.get_condition()

        self.assertEqual(condition, 100)

    def test_reset_any_reduction_factor_half(self):
        cond = ConditionIndicator(
            perfect=100, failed=0, pf_curve="linear", pf_interval=10
        )
        cond.set_condition(50)
        cond.reset_any(target=0.5, method="reduction_factor", axis="condition")
        condition = cond.get_condition()

        self.assertEqual(condition, 75)

    def test_reset_any_reduction_factor_none(self):
        cond = ConditionIndicator(
            perfect=100, failed=0, pf_curve="linear", pf_interval=10
        )

        cond.set_condition(50)
        cond.reset_any(target=0, method="reduction_factor", axis="condition")
        condition = cond.get_condition()

        self.assertEqual(condition, 50)

    # ************** Test the accumulation functions ******************

    def test_accumulate_time(self):
        self.assertEqual(False, False)

    def test_update(self):

        test_data_1 = copy.deepcopy(fixtures.condition_data["fast_degrading"])
        test_data_1["name"] = "FD"
        test_data_1["perfect"] = 99
        test_data_1["pf_std"] = 0.25
        test_data_2 = copy.deepcopy(fixtures.condition_data["fast_degrading"])

        c1 = ConditionIndicator.from_dict(test_data_1)
        c2 = ConditionIndicator.from_dict(test_data_2)

        c1.update_from_dict({"name": "fast_degrading", "perfect": 100, "pf_std": 0.5})

        # self.assertEqual(c1.__dict__, c2.__dict__)
        self.assertEqual(c1.name, c2.name)
        self.assertEqual(c1.perfect, c2.perfect)
        self.assertEqual(c1.pf_std, c2.pf_std)

    def test_update_error(self):

        c = ConditionIndicator.from_dict()
        update = {"alpha": 10, "beta": 5}

        self.assertRaises(KeyError, c.update_from_dict, update)


if __name__ == "__main__":
    unittest.main()