
import unittest
import numpy as np

import utils

from pof.condition import Condition

class TestCondition(unittest.TestCase):

    def setUp(self):
        self.cld = Condition(100,0,'linear', [-10])
        self.cli = Condition(0,100,'linear', [10])

    def test_class_imports_correctly(self):
        self.assertTrue(True)

    def test_class_instantiate(self):
        c = Condition()
        c.__dict__
        self.assertTrue(True)
        
    # Check whole degradation
    # test_starts_perfect_ends_perfect
    # test_starts_perfect_ends_partial
    # test_starts_perfect_ends_failed
    # test_starts_partial_ends_partial
    # test_starts_partial_ends_partial
    # test_starts_partial_ends_failed

    # test_perfect_prior_to_start
    # test_partial_prior_to_start

    # *********** Test the condition limits ******************
    def test_condition_starts_zero(self):
        d = Condition(perfect=0, failed=100, pf_curve='linear', pf_curve_params =[10])
        self.assertEqual(d.current(), 0)

    def test_condition_starts_zero_does_not_breach_limit(self):
        d = Condition(perfect=0, failed=100, pf_curve='linear', pf_curve_params =[10])
        d.sim(100)
        self.assertEqual(d.current(), 100)

    def test_condition_starts_positive(self):
        d = Condition(perfect=100, failed=0, pf_curve='linear', pf_curve_params =[-10])
        self.assertEqual(d.current(), 100)

    def test_condition_starts_positive_does_not_breach_limit(self):
        d = Condition(perfect=100, failed=0, pf_curve='linear', pf_curve_params =[-10])
        d.sim(100)
        self.assertEqual(d.current(), 0)


    # ********** Test get_condition_profile **********

    # early start

    def test_get_condition_profile_early_start_early_stop(self):
        c = Condition(100,50,'linear',[-1])
        expected = np.concatenate((np.full(10,100), np.linspace(100,90,11)))
        cp = c.get_condition_profile(t_start=-10, t_stop=10)
        np.testing.assert_array_equal(cp, expected)

    def test_get_condition_profile_early_start_late_stop(self):
        c = Condition(100,50,'linear',[-1])
        expected = np.concatenate((np.full(10,100), np.linspace(100,50,51), np.full(50,50)))
        cp = c.get_condition_profile(t_start=-10, t_stop=100)
        np.testing.assert_array_equal(cp, expected)

    def test_get_condition_profile_early_start_no_stop(self):
        c = Condition(100,50,'linear',[-1])
        expected = np.concatenate((np.full(10,100), np.linspace(100,50,51)))
        cp = c.get_condition_profile(t_start=-10)
        np.testing.assert_array_equal(cp, expected)

    # existing condition

    def test_get_condition_profile_existing_condition_early_start_early_stop(self):
        expected = np.concatenate((np.full(10,90), np.linspace(90,70,21)))
        c = Condition(100,50,'linear',[-1])
        c.set_condition(90)
        cp = c.get_condition_profile(t_start=-10, t_stop = 20)
        np.testing.assert_array_equal(cp, expected)

    def test_get_condition_profile_existing_condition_early_start_late_stop(self):
        expected = np.concatenate((np.full(10,90), np.linspace(90,50,41), np.full(60,50)))
        c = Condition(100,50,'linear',[-1])
        c.set_condition(90)
        cp = c.get_condition_profile(t_start=-10, t_stop = 100)
        np.testing.assert_array_equal(cp, expected)

    def test_get_condition_profile_existing_condition_early_start_no_stop(self):
        expected = np.concatenate((np.full(10,90), np.linspace(90,50,41)))
        c = Condition(100,50,'linear',[-1])
        c.set_condition(90)
        cp = c.get_condition_profile(t_start=-10)
        np.testing.assert_array_equal(cp, expected)

    def test_get_condition_profile_existing_condition_v_early_start_v_late_stop(self):
        expected = np.concatenate((np.full(10, 90), np.linspace(90,50,41), np.full(60,50)))
        c = Condition(100,50,'linear',[-1]) 
        c.set_condition(90)
        cp = c.get_condition_profile(t_start=-10, t_stop=100)
        np.testing.assert_array_equal(cp, expected)

    def test_get_condition_profile_existing_condition_v_late_start_v_late_stop(self):
        expected = np.full(41,50)
        c = Condition(100,50,'linear',[-1]) 
        c.set_condition(50)
        cp = c.get_condition_profile(t_start=60, t_stop=100)
        np.testing.assert_array_equal(cp, expected)

    # late start

    def test_get_condition_profile_late_start_early_stop(self):
        c = Condition(100,50,'linear',[-1])
        expected = np.linspace(95,90,6)
        cp = c.get_condition_profile(t_start=5, t_stop=10)
        np.testing.assert_array_equal(cp, expected)

    def test_get_condition_profile_late_start_late_stop(self):
        c = Condition(100,50,'linear',[-1])
        expected = np.concatenate((np.linspace(95,50,46), np.full(50, 50)))
        cp = c.get_condition_profile(t_start=5, t_stop=100)
        np.testing.assert_array_equal(cp, expected)

    def test_get_condition_profile_late_start_no_stop(self):
        c = Condition(100,50,'linear',[-1])
        expected = np.linspace(95,50,46)
        cp = c.get_condition_profile(t_start=5)
        np.testing.assert_array_equal(cp, expected)

    # no start

    def test_get_condition_profile_no_start_early_stop(self):
        c = Condition(100,50,'linear',[-1])
        expected = np.linspace(100,90,11)
        cp = c.get_condition_profile(t_stop=10)
        np.testing.assert_array_equal(cp, expected)

    def test_get_condition_profile_no_start_late_stop(self):
        c = Condition(100,50,'linear',[-1])
        expected = np.concatenate((np.linspace(100,50,51), np.full(50,50)))
        cp = c.get_condition_profile( t_stop=100)
        np.testing.assert_array_equal(cp, expected)

    def test_get_condition_profile_no_start_no_stop(self):
        c = Condition(100,50,'linear',[-1])
        expected = np.linspace(100,50,51)
        cp = c.get_condition_profile()
        np.testing.assert_array_equal(cp, expected)        


    # very early start

    def test_get_condition_profile_v_early_start_early_stop(self):
        c = Condition(100,50,'linear',[-1])
        expected = np.concatenate((np.full(100,100), np.linspace(100,90,11)))
        cp = c.get_condition_profile(t_start=-100, t_stop=10)
        np.testing.assert_array_equal(cp, expected)

    def test_get_condition_profile_v_early_start_late_stop(self):
        c = Condition(100,50,'linear',[-1])
        expected = np.concatenate((np.full(100,100), np.linspace(100,50,51), np.full(50,50)))
        cp = c.get_condition_profile(t_start=-100, t_stop=100)
        np.testing.assert_array_equal(cp, expected)

    def test_get_condition_profile_v_early_start_no_stop(self):
        c = Condition(100,50,'linear',[-1])
        expected = np.concatenate((np.full(100,100), np.linspace(100,50,51)))
        cp = c.get_condition_profile(t_start=-100)
        np.testing.assert_array_equal(cp, expected)

    def test_get_condition_profile_v_early_start_v_early_stop(self):
        expected = np.full(91,100)
        c = Condition(100,50,'linear',[-1])
        cp = c.get_condition_profile(t_start=-100, t_stop=-10)
        np.testing.assert_array_equal(cp, expected)

    # very late start

    def test_get_condition_profile_v_late_start_early_stop(self):
        c = Condition(100,50,'linear',[-1])
        expected = [90]
        cp = c.get_condition_profile(t_start=20, t_stop=10)
        np.testing.assert_array_equal(cp, expected)

    def test_get_condition_profile_v_late_start_late_stop(self):
        c = Condition(100,50,'linear',[-1])
        expected = [50]
        cp = c.get_condition_profile(t_start=110, t_stop=100)
        np.testing.assert_array_equal(cp, expected)

    def test_get_condition_profile_v_late_start_no_stop(self):
        c = Condition(100,50,'linear',[-1])
        expected = [50]
        cp = c.get_condition_profile(t_start=60)
        np.testing.assert_array_equal(cp, expected)


    # ********** Test set_condition **************** TODO add condition checks as well as time

    def test_set_condition_pf_decreasing_above_limit(self):
        c = Condition(100,50,'linear',[-10])
        c.set_condition(150)
        self.assertEqual(c.t_condition, 0)

    def test_set_condition_pf_decreasing_below_limit(self):
        c = Condition(100,50,'linear',[-10])
        c.set_condition(0)
        self.assertEqual(c.t_condition, 5)

    def test_set_condition_pf_decreasing_in_limit(self):
        c = Condition(100,50,'linear',[-10])
        c.set_condition(70)
        self.assertEqual(c.t_condition, 3)

    def test_set_condition_pf_increasing_above_limit(self):
        c = Condition(50,100,'linear',[10], decreasing = False)
        c.set_condition(150)
        self.assertEqual(c.t_condition, 5)

    def test_set_condition_pf_increasing_below_limit(self):
        c = Condition(50,100,'linear',[10], decreasing = False)
        c.set_condition(0)
        self.assertEqual(c.t_condition, 0)

    def test_set_condition_pf_increasing_in_limit(self):
        c = Condition(50,100,'linear',[10], decreasing = False)
        c.set_condition(70)
        self.assertEqual(c.t_condition, 2)


    # **************** Test set_t_condition ***************

    def test_set_t_condition_above_limit(self):
        c = Condition(100,0,'linear', [-10])
        c.set_t_condition(150)
        self.assertEqual(c.t_condition, 10)

    def test_set_t_condition_below_limit(self):
        c = Condition(100,0,'linear', [-10])
        c.set_t_condition(-10)
        self.assertEqual(c.t_condition, 0)

    def test_set_t_condition_in_limit(self):
        c = Condition(100,0,'linear', [-10])
        c.set_t_condition(5)
        self.assertEqual(c.t_condition, 5)

    # **************** Test limit_reached *****************

    def test_is_failed_pf_decreasing_at_threshold (self):
        c = Condition(100,0,'linear', [-10])
        c.threshold_failure = 50
        c.set_condition(50)
        self.assertTrue(c.is_failed())

    def test_is_failed_pf_decreasing_exceeds_threshold (self):
        c = Condition(100,0,'linear', [-10])
        c.threshold_failure = 50
        c.set_condition(30)
        self.assertTrue(c.is_failed())

    def test_is_failed_pf_decreasing_within_threshold (self):
        c = Condition(100,0,'linear', [-10])
        c.threshold_failure = 50
        c.set_condition(70)
        self.assertFalse(c.is_failed())
    
    def test_is_failed_pf_increasing_at_threshold (self):
        c = Condition(0,100,'linear', [10], decreasing = False)
        c.threshold_failure = 50
        c.set_condition(50)
        self.assertTrue(c.is_failed())

    def test_is_failed_pf_increasing_exceeds_threshold (self):
        c = Condition(0,100,'linear', [10], decreasing = False)
        c.threshold_failure = 50
        c.set_condition(70)
        self.assertTrue(c.is_failed())

    def test_is_failed_pf_increasing_within_threshold (self):
        c = Condition(0,100,'linear', [10], decreasing = False)
        c.threshold_failure = 50
        c.set_condition(30)
        self.assertFalse(c.is_failed())

    # **************** Test the measuring functions

    def test_detectable_condition_starts_zero_current_above_detection_threshold_zero(self):
        d = Degradation(perfect=0, limit=100, pf_curve='linear', pf_curve_params =[-10])
        d.condition_detectable = 0
        d.set_condition(30)
        self.assertTrue(d.detectable())

    def test_detectable_condition_starts_zero_current_at_detection_threshold_zero(self):
        d = Degradation(perfect=0, limit=100, pf_curve='linear', pf_curve_params =[-10])
        d.condition_detectable = 0
        d.set_condition(0)
        self.assertTrue(d.detectable())

    def test_detectable_condition_starts_zero_current_above_detection_threshold_positive(self):
        d = Degradation(perfect=0, limit=100, pf_curve='linear', pf_curve_params =[-10])
        d.condition_detectable = 20
        d.set_condition(30)
        self.assertTrue(d.detectable())

    def test_detectable_condition_starts_zero_current_at_detection_threshold_positive(self):
        d = Degradation(perfect=0, limit=100, pf_curve='linear', pf_curve_params =[-10])
        d.condition_detectable = 20
        d.set_condition(20)
        self.assertTrue(d.detectable())

    def test_detectable_condition_starts_positive_current_below_detection_threshold_positive(self):
        d = Degradation(perfect=100, limit=0, pf_curve='linear', pf_curve_params =[-10])
        d.condition_detectable = 80
        d.set_condition(d.condition_perfect)
        self.assertFalse(d.detectable())

    def test_detectable_condition_starts_positive_current_above_detection_threshold_zero(self):
        d = Degradation(perfect=100, limit=0, pf_curve='linear', pf_curve_params =[-10])
        d.condition_detectable = 80
        d.set_condition(90)
        self.assertFalse(d.detectable())

    def test_detectable_condition_starts_positive_current_at_detection_threshold_zero(self):
        d = Degradation(perfect=100, limit=0, pf_curve='linear', pf_curve_params =[-10])
        d.condition_detectable = 80
        d.set_condition(0)
        self.assertTrue(d.detectable())

    def test_detectable_condition_starts_positive_current_above_detection_threshold_positive(self):
        d = Degradation(perfect=100, limit=0, pf_curve='linear', pf_curve_params =[-10])
        d.condition_detectable = 80
        d.set_condition(70)
        self.assertTrue(d.detectable())

    def test_detectable_condition_starts_positive_current_at_detection_threshold_positive(self):
        d = Degradation(perfect=100, limit=0, pf_curve='linear', pf_curve_params =[-10])
        d.condition_detectable = 80
        d.set_condition(80)
        self.assertTrue(d.detectable())


    # **************** Test the reset functions ***********************

    def test_reset(self):
        c = Condition(100,0,'linear', [-10], decreasing = True)
        c.sim(10)
        c.reset()
        self.assertEqual(c.current(),100)

    # **************** Test the reset_any functions ***********************
    
    def test_reset_any_reduction_factor_all(self):
        c = Condition(100,0,'linear', [-10], decreasing = True)
        c.set_condition(50)
        c.reset_any(target=1, method='reduction_factor', axis='condition')
        self.assertEqual(c.condition, 100)

    def test_reset_any_reduction_factor_half(self):
        c = Condition(100,0,'linear', [-10], decreasing = True)
        c.set_condition(50)
        c.reset_any(target=0.5, method='reduction_factor', axis='condition')
        self.assertEqual(c.condition, 75)

    def test_reset_any_reduction_factor_none(self):
        c = Condition(100,0,'linear', [-10], decreasing = True)
        c.set_condition(50)
        c.reset_any(target=0, method='reduction_factor', axis='condition')
        self.assertEqual(c.condition, 50)

    

    # ************** Test the accumulation functions ******************

    def test_accumulate_time(self):
        self.assertEqual(False, False)

if __name__ == '__main__':
    unittest.main()
