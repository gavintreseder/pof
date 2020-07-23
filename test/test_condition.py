

import unittest

from pof.condition import Condition

class TestCondition(unittest.TestCase):

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
        d = Degradation(perfect=0, limit=100, cond_profile_type='linear', cond_profile_params =[10])
        self.assertEqual(d.current(), 0)

    def test_condition_starts_zero_does_not_breach_limit(self):
        d = Degradation(perfect=0, limit=100, cond_profile_type='linear', cond_profile_params =[10])
        d.sim(100)
        self.assertEqual(d.current(), 100)

    def test_condition_starts_positive(self):
        d = Degradation(perfect=100, limit=0, cond_profile_type='linear', cond_profile_params =[-10])
        self.assertEqual(d.current(), 100)

    def test_condition_starts_positive_does_not_breach_limit(self):
        d = Degradation(perfect=100, limit=0, cond_profile_type='linear', cond_profile_params =[-10])
        d.sim(100)
        self.assertEqual(d.current(), 0)


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

    # **************** Test the measuring functions

    def test_detectable_condition_starts_zero_current_above_detection_threshold_zero(self):
        d = Degradation(perfect=0, limit=100, cond_profile_type='linear', cond_profile_params =[-10])
        d.condition_detectable = 0
        d.set_condition(30)
        self.assertTrue(d.detectable())

    def test_detectable_condition_starts_zero_current_at_detection_threshold_zero(self):
        d = Degradation(perfect=0, limit=100, cond_profile_type='linear', cond_profile_params =[-10])
        d.condition_detectable = 0
        d.set_condition(0)
        self.assertTrue(d.detectable())

    def test_detectable_condition_starts_zero_current_above_detection_threshold_positive(self):
        d = Degradation(perfect=0, limit=100, cond_profile_type='linear', cond_profile_params =[-10])
        d.condition_detectable = 20
        d.set_condition(30)
        self.assertTrue(d.detectable())

    def test_detectable_condition_starts_zero_current_at_detection_threshold_positive(self):
        d = Degradation(perfect=0, limit=100, cond_profile_type='linear', cond_profile_params =[-10])
        d.condition_detectable = 20
        d.set_condition(20)
        self.assertTrue(d.detectable())

    def test_detectable_condition_starts_positive_current_below_detection_threshold_positive(self):
        d = Degradation(perfect=100, limit=0, cond_profile_type='linear', cond_profile_params =[-10])
        d.condition_detectable = 80
        d.set_condition(d.condition_perfect)
        self.assertFalse(d.detectable())

    def test_detectable_condition_starts_positive_current_above_detection_threshold_zero(self):
        d = Degradation(perfect=100, limit=0, cond_profile_type='linear', cond_profile_params =[-10])
        d.condition_detectable = 80
        d.set_condition(90)
        self.assertFalse(d.detectable())

    def test_detectable_condition_starts_positive_current_at_detection_threshold_zero(self):
        d = Degradation(perfect=100, limit=0, cond_profile_type='linear', cond_profile_params =[-10])
        d.condition_detectable = 80
        d.set_condition(0)
        self.assertTrue(d.detectable())

    def test_detectable_condition_starts_positive_current_above_detection_threshold_positive(self):
        d = Degradation(perfect=100, limit=0, cond_profile_type='linear', cond_profile_params =[-10])
        d.condition_detectable = 80
        d.set_condition(70)
        self.assertTrue(d.detectable())

    def test_detectable_condition_starts_positive_current_at_detection_threshold_positive(self):
        d = Degradation(perfect=100, limit=0, cond_profile_type='linear', cond_profile_params =[-10])
        d.condition_detectable = 80
        d.set_condition(80)
        self.assertTrue(d.detectable())


    # **************** Test the reset functions ***********************

    def test_reset(self):
        d = Degradation(perfect=100, limit=0, cond_profile_type='linear', cond_profile_params =[-10])
        d.sim(10)
        d.reset()
        self.assertEqual(d.current(),100)

    def test_reset_(self):
        d = Degradation(perfect=100, limit=0, cond_profile_type='linear', cond_profile_params =[-10])
        d.sim(10)
        d.reset()
        self.assertEqual(d.current(),100)

    # ************** Test the accumulation functions ******************

    def test_accumulate_time(self):
        self.assertEqual(False)

if __name__ == '__main__':
    unittest.main()
