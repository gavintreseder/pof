import unittest
from unittest.mock import Mock
import copy
from random import randint

import numpy as np

import fixtures
import testconfig
from test_load import TestPofBase
from pof.indicator import ConditionIndicator
import pof.demo as demo


# TODO fix up data sources from a single location

cid = dict(
    name="test",
    perfect=100,
    failed=0,
    pf_curve="linear",
    pf_interval=100,
    pf_std=None,
)

params_perfect_failed_int = [(0, 100), (100, 0), (50, 100)]
params_perfect_failed_bool = [(True, False), (False, True)]


class TestConditionIndicator(TestPofBase, unittest.TestCase):
    def setUp(self):
        super().setUp()

        # TestInstantiate
        self._class = ConditionIndicator

        # TestLoadFromdict
        self._data_valid = dict(name="TestConditionIndicator", pf_curve="step")
        self._data_invalid_values = [{"pf_curve": "invalid_value"}]
        self._data_invalid_types = [{"invalid_type": "invalid_type"}]

        cond_data = demo.condition_data["instant"]
        self.cond = ConditionIndicator.load(cond_data)

    # Next test

    def test_sim_timeline_two_causes(self):

        cond = ConditionIndicator(perfect=100, failed=0)

        cond.sim_timeline(t_stop=200, name="cause_1")
        cond.sim_timeline(t_stop=200, name="cause_2")

        timeline_1 = cond.get_timeline(name="cause_1")
        timeline_2 = cond.get_timeline(name="cause_2")

    # ********************* Test sim_timeline *********************

    def test_sim_timeline(self):
        cond = ConditionIndicator.from_dict(cid)
        cond.sim_timeline(t_stop=100)
        self.assertIsNotNone(cond)

    def test_sim_timeline_no_data(self):
        cond = ConditionIndicator.from_dict(cid)
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

    # TODO test the rest of the limits wiht nose@param

    def test_condition_starts_zero(self):
        cond = ConditionIndicator(
            perfect=0, failed=100, pf_curve="linear", pf_interval=10
        )
        self.assertEqual(cond.get_condition(), 0)

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
        # TODO revisit which behaviour would be better
        # expected = np.concatenate((np.full(10, 90), np.linspace(90, 50, 41)))
        expected = np.concatenate(
            (np.full(10, 90), np.linspace(90, 50, 41), np.full(10, 50))
        )
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

    # **************** Test the reset functions ***********************

    def test_reset(self):
        cond = ConditionIndicator(
            perfect=100, failed=0, pf_curve="linear", pf_interval=10
        )
        cond.set_condition(50)
        cond.reset()
        self.assertEqual(cond.get_condition(), 100)

    def test_reset_after_executing_methods(self):
        # Arrange
        expected = ConditionIndicator(
            perfect=100, failed=0, pf_curve="linear", pf_interval=10
        )

        cond = ConditionIndicator(
            perfect=100, failed=0, pf_curve="linear", pf_interval=10
        )

        cond.sim_timeline(100)
        cond.reset_any(target=0.5, method="reduction_factor", axis="condition")

        # Act
        cond.reset()

        # Assert
        self.assertEqual(cond, expected)

    def test_reset_for_next_sim(self):
        """Check that the appropriate areas are reset correctly for the next simulation"""
        param_set_up = [(100, 0), (0, 100)]
        param_pf_curve = ["linear", "step"]
        param_initial = [0, 25, 50, 75, 100]
        iterations = 10

        for pf_curve in param_pf_curve:
            for perfect, failed in param_set_up:
                for initial in param_initial:

                    # with self.subTest():
                    # Arrange
                    ind = ConditionIndicator(
                        perfect=perfect, failed=failed, pf_curve=pf_curve
                    )
                    ind.set_initial(initial)
                    expected = abs(perfect - initial)

                    # Act
                    ind.mc_timeline(t_end=100, t_start=0, n_iterations=iterations)
                    ind.reset_for_next_sim()

                    # Assert
                    self.assertEqual(
                        ind.get_accumulated(),
                        expected,
                        "accumulated should equal intial",
                    )

    # **************** Test the _set_accumulated  ***********************

    def test__set_accumulated(self):
        param_list = [(100, 0), (0, 100)]
        param_initial = [0, 25, 50, 75, 100]
        param_accumulated = [0, 25, 50, 75, 100]
        param_name = [None, "cause_1"]

        for perfect, failed in param_list:
            for initial in param_initial:
                for accumulated in param_accumulated:
                    for name in param_name:

                        # with self.subTest():

                        # Arrange
                        ind = ConditionIndicator(
                            perfect=perfect,
                            failed=failed,
                            pf_curve="linear",
                            initial=initial,
                            pf_interval=100,
                        )

                        expected = min(
                            accumulated + abs(perfect - initial),
                            abs(perfect - failed),
                        )

                        # Act
                        ind._set_accumulated(accumulated=accumulated, name=name)
                        result = ind.get_accumulated()

                        # Assert
                        self.assertEqual(result, expected)

    # TODO add robust tests for accumulated

    # def test_sim_timeline_with_accumulated(self):
    #     """Check that the appropriate areas are reset correctly for the next simulation"""
    #     param_set_up = [(100, 0), (0, 100)]
    #     param_pf_curve = ["linear"]
    #     param_initial = [0, 25, 50, 75, 100]
    #     param_accumulated = [0, 25, 50, 75, 100]
    #     param_name = [None, "cause_1"]
    #     param_permanent = [False, True]

    #     for pf_curve in param_pf_curve:
    #         for perfect, failed in param_set_up:
    #             for initial in param_initial:
    #                 for accumulated in param_accumulated:
    #                     for name in param_name:
    #                         for permanent in param_permanent:
    #                             # with self.subTest():
    #                             # Arrange
    #                             ind = ConditionIndicator(
    #                                 perfect=perfect,
    #                                 failed=failed,
    #                                 pf_curve=pf_curve,
    #                                 initial=initial,
    #                                 pf_interval=100,
    #                             )

    #                             expected = abs(perfect - initial)

    #                             # Act
    #                             ind.sim_timeline(t_start=0, t_stop=0)
    #                             ind._reset_accumulated(
    #                                 accumulated=accumulated,
    #                                 name=name,
    #                                 permanent=permanent,
    #                             )

    #                             # Assert
    #                             self.assertEqual(
    #                                 ind.get_accumulated(),
    #                                 expected,
    #                                 "accumulated should equal initial",
    #                             )

    def test_reset_for_next_sim_complex(self):
        """Check that the appropriate areas are reset correctly for the next simulation"""
        param_set_up = [(100, 0), (0, 100)]
        param_pf_curve = ["linear", "step"]
        param_initial = [0, 25, 50, 75, 100]
        param_accumulated = [0, 25, 50, 75, 100]
        param_name = [None, "cause_1"]

        for perfect, failed in param_set_up:
            for pf_curve in param_pf_curve:
                for initial in param_initial:
                    for accumulated in param_accumulated:
                        for name in param_name:

                            # with self.subTest():
                            # Arrange
                            ind = ConditionIndicator(
                                perfect=perfect,
                                failed=failed,
                                pf_curve=pf_curve,
                                initial=initial,
                            )

                            expected = abs(perfect - initial)

                            # Act
                            ind.reset_for_next_sim()

                            # Assert
                            self.assertEqual(
                                ind.get_accumulated(),
                                expected,
                                "accumulated should equal intial",
                            )

    # **************** Test the reset_any functions ***********************

    def test_reset_any_reduction_factor_all(self):

        param_list = [(100, 0, 20), (0, 100, 80), (110, 10, 30), (10, 110, 90)]

        for perfect, failed, current in param_list:
            # with self.subTest():
            # Arrange
            cond = ConditionIndicator(
                perfect=perfect, failed=failed, pf_curve="linear", pf_interval=10
            )
            cond.set_condition(current)

            for expected in range(2):
                # Act
                cond.reset_any(target=1, method="reduction_factor", axis="condition")
                condition = cond.get_condition()

                # Assert
                self.assertEqual(condition, perfect)

    def test_reset_any_reduction_factor_half(self):

        param_list = [
            (100, 0, 20, [60, 80, 90, 95]),
            (0, 100, 80, [40, 20, 10, 5]),
            (110, 10, 30, [70, 90, 100, 105]),
            (10, 110, 90, [50, 30, 20, 15]),
        ]

        for perfect, failed, current, results in param_list:
            # with self.subTest():
            # Arrange
            cond = ConditionIndicator(
                perfect=perfect, failed=failed, pf_curve="linear", pf_interval=10
            )
            cond.set_condition(current)

            for expected in results:
                # Act
                cond.reset_any(target=0.5, method="reduction_factor", axis="condition")
                condition = cond.get_condition()

                # Assert
                self.assertEqual(condition, expected)

    def test_reset_any_reduction_factor_none(self):

        param_list = [
            (100, 0, 20),
            (0, 100, 80),
            (110, 10, 30),
            (10, 110, 90),
        ]

        for perfect, failed, current in param_list:
            # with self.subTest():
            # Arrange
            cond = ConditionIndicator(
                perfect=perfect, failed=failed, pf_curve="linear", pf_interval=10
            )
            cond.set_condition(current)

            for expected in range(2):
                # Act
                cond.reset_any(target=0, method="reduction_factor", axis="condition")
                condition = cond.get_condition()

                # Assert
                self.assertEqual(condition, current)

    # ************** Test the accumulation functions ******************

    def test_accumulate_time(self):
        self.assertEqual(False, False)

    def test_update(self):

        test_data_1 = copy.deepcopy(fixtures.condition_data["fast_degrading"])
        test_data_1["name"] = "FD"
        test_data_1["pf_std"] = 0.25
        test_data_2 = copy.deepcopy(fixtures.condition_data["fast_degrading"])

        c1 = ConditionIndicator.from_dict(test_data_1)
        c2 = ConditionIndicator.from_dict(test_data_2)

        c1.update({"name": "fast_degrading", "pf_std": 0.5})

        self.assertEqual(c1, c2)

    def test_update_error(self):

        test_data = copy.deepcopy(fixtures.condition_data["fast_degrading"])

        c = ConditionIndicator.from_dict(test_data)
        update = {"alpha": 10, "beta": 5}

        self.assertRaises(KeyError, c.update, update)

    def test_agg_timelines_no_cause(self):
        """
        Check that the timelines aggregate correctly and do not exceed the limits
        """
        expected = np.reshape(
            np.concatenate([np.linspace(100, 0, 101), np.full(100, 0)]), (1, 201)
        )

        cond = ConditionIndicator(
            pf_curve="linear", perfect=100, failed=0, pf_interval=100
        )

        cond.sim_timeline(t_start=0, t_stop=200)
        cond.save_timeline()
        cond.save_timeline()
        agg_timeline = cond.agg_timelines()

        np.testing.assert_array_equal(agg_timeline, expected)

    def test_agg_timelines_mutliple_timelines(self):
        """
        Check that the timelines aggregate correctly and do not exceed the limits
        """
        expected = np.tile(
            np.concatenate(
                [np.linspace(100, 51, 50), np.linspace(50, 0, 26), np.full(125, 0)]
            ),
            (2, 1),
        )

        cond = ConditionIndicator(
            pf_curve="linear", perfect=100, failed=0, pf_interval=100
        )

        cond.sim_timeline(name="cause_1", t_start=0, t_stop=200)
        cond.sim_timeline(name="cause_2", t_start=-50, t_stop=150)
        cond.save_timeline(1)
        cond.save_timeline(2)

        agg_timeline = cond.agg_timelines()

        np.testing.assert_array_equal(agg_timeline, expected)

    def test_agg_timeline_mutltiple_causes(self):
        """
        Check that the causes aggregate correctly and do not exceed the limits
        """
        expected = np.concatenate(
            [np.linspace(100, 51, 50), np.linspace(50, 0, 26), np.full(125, 0)]
        )

        cond = ConditionIndicator(
            pf_curve="linear", perfect=100, failed=0, pf_interval=100
        )

        cond.sim_timeline(name="cause_1", t_start=0, t_stop=200)
        cond.sim_timeline(name="cause_2", t_start=-50, t_stop=150)
        cond.save_timeline()

        agg_timeline = cond.agg_timeline()

        np.testing.assert_array_equal(agg_timeline, expected)

    def test_expected_condition_one_timeline(self):
        NotImplemented  # TODO

    def test_expected_condition(self):

        cond = ConditionIndicator(
            pf_curve="linear", perfect=100, failed=0, pf_interval=100
        )

        for i in range(100):
            pf_interval = randint(100, 300)
            cond.sim_timeline(
                name="cause_1", t_start=0, t_stop=200, pf_interval=pf_interval
            )
            cond.save_timeline()

        ec = cond.expected_condition()

    def test_sim_failure_timeline(self):
        """Checks that a failure timeline returns the correct values"""
        param_t_delay = [0, 10]
        param_pf_curve = ["linear", "step"]
        param_list = [
            (100, 0, 1),
            (100, 0, 50),
            (100, 0, 99),
            (0, 100, 1),
            (0, 100, 50),
            (0, 100, 99),
            (False, True, True),
            (True, False, False),
        ]
        pf_interval = 100

        for t_delay in param_t_delay:
            for perfect, failed, threshold_failure in param_list:
                for pf_curve in param_pf_curve:
                    # expected
                    if pf_curve == "linear":
                        n_ok = int(
                            abs(perfect - threshold_failure)
                            / abs(perfect - failed)
                            * pf_interval
                        )
                    elif pf_curve == "step":
                        n_ok = pf_interval
                    else:
                        self.fail()

                    n_failure = pf_interval - n_ok + 1
                    expected = np.concatenate(
                        [np.full(n_ok, False), np.full(n_failure, True)]
                    )
                    expected = expected[t_delay:]

                    # with self.subTest():
                    # Arrange
                    ind = ConditionIndicator(
                        perfect=perfect,
                        failed=failed,
                        threshold_failure=threshold_failure,
                        pf_interval=pf_interval,
                        pf_curve=pf_curve,
                    )

                    # Act
                    ind.sim_timeline(t_stop=100)
                    ft = ind.sim_failure_timeline(t_stop=100, t_delay=t_delay)

                    # Assert
                    np.testing.assert_array_equal(ft, expected)


class TestPoleSafetyFactor(unittest.TestCase):
    def test_sim_timeline(self):
        NotImplemented


if __name__ == "__main__":
    unittest.main()
