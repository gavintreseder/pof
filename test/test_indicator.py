
import unittest
from unittest.mock import Mock
import numpy as np

import utils

from pof.indicator import ConditionIndicator
import pof.demo as demo


cid = dict(
    name = 'test',
    perfect = 100,
    failed = 0,
    pf_curve = 'linear',
    pf_interval = 100,
    pf_std = None,
)

class TestConditionIndicator(unittest.TestCase):
    def test_class_imports_correctly(self):
        self.assertTrue(True)


    def test_instantiate(self):
        try:
            ci = ConditionIndicator()
            self.assertIsNotNone(ci)
        except ValueError:
            self.fail('ValueError returned')
        except:
            self.fail('Unknown error')


    def test_instantiate_with_data(self):
        try:
            ci = ConditionIndicator(name = 'test_name')
            self.assertIsNotNone(ci)
        except ValueError:
            self.fail('ValueError returned')
        except:
            self.fail('Unknown error')

    def test_from_dict(self):
        try:
            ci = ConditionIndicator.from_dict()
            self.assertIsNotNone(ci)
        except ValueError:
            self.fail('ValueError returned')
        except:
            self.fail('Unknown error')

    def test_from_dict_value_error_exits(self):

        false_data = dict(pf_curve = 'incorrect_value')

        with self.assertRaises(ValueError) as cm:
            ci = ConditionIndicator.from_dict(false_data)

    def test_from_dict_with_data(self):
        try:
            ci = ConditionIndicator.from_dict(demo.condition_data['instant'])
            self.assertIsNotNone(ci)
        except ValueError:
            self.fail('ValueError returned')
        except:
            self.fail('Unknown error')
    
    # ********************* Test sim_timeline *********************

    def test_sim_timeline(self):
        ci = ConditionIndicator()
        ci.sim_timeline(t_stop=100)

    def test_sim_timeline_no_data(self):
        ci = ConditionIndicator()
        ci.sim_timeline()
        self.assertIsNotNone(ci)

    def test_sim_timeline_no_data_passed(self):
        ci = ConditionIndicator.from_dict(cid)
        ci.sim_timeline()

    def test_sim_timeline_order_none_then_data(self):

        mock = Mock()
        mock(100)
        mock(t_stop=100, pf_interval=10, name='cause_1')
        mock(t_stop=100, pf_interval=20, name='cause_1')


        expected = {
            0 : np.linspace(100,0,101),
            1 : np.concatenate([np.linspace(100,0,11), np.full(90,0)]),
            2 : np.concatenate([np.linspace(100,0,21), np.full(80,0)]),
        }

        ci = ConditionIndicator.from_dict(cid)

        for k, v in expected.items():
            # Unpack the test data
            args, kwargs = mock.call_args_list[k]
            name = mock.call_args_list[k].kwargs.get('name', None)
            ci.sim_timeline(*args, **kwargs) # TODO check why timeline isn't updating
            np.testing.assert_array_equal(ci.get_timeline(name), v, err_msg='Failed with %s' %(str(mock.call_args_list[k])))
        

    def test_sim_timeline_order_data_then_none(self):
        mock = Mock()
        mock(t_stop=100, pf_interval=10, name='cause_1')
        mock(t_stop=100, pf_interval=20, name='cause_1')
        mock(t_stop=100, pf_interval=40, name='cause_2')
        mock(100)


        expected = {
            0 : np.concatenate([np.linspace(100,0,11), np.full(90,0)]),
            1 : np.concatenate([np.linspace(100,0,21), np.full(80,0)]),
            2 : np.concatenate([np.linspace(100,0,41), np.full(60,0)]),
            3 : np.linspace(100,0,101),
        }

        ci = ConditionIndicator.from_dict(cid)

        for k, v in expected.items():
            # Unpack the test data
            args, kwargs = mock.call_args_list[k]
            name = mock.call_args_list[k].kwargs.get('name', None)
            ci.sim_timeline(*args, **kwargs) # TODO check why timeline isn't updating
            np.testing.assert_array_equal(ci.get_timeline(name), v, err_msg='Failed with %s' %(str(mock.call_args_list[k])))
        


    # *********** Test the condition limits ******************
    def test_condition_starts_zero(self):
        ci = ConditionIndicator(perfect=0, failed=100, pf_curve='linear', pf_interval=10)
        self.assertEqual(ci.get_condition(), 0)

    def test_condition_starts_zero_does_not_breach_limit(self):
        ci = ConditionIndicator(perfect=0, failed=100, pf_curve='linear', pf_interval=10)
        ci.sim(100)
        self.assertEqual(ci.get_condition(), 100)

    def test_condition_starts_positive(self):
        ci = ConditionIndicator(perfect=100, failed=0, pf_curve='linear', pf_interval=10)
        self.assertEqual(ci.get_condition(), 100)

    def test_condition_starts_positive_does_not_breach_limit(self):
        ci = ConditionIndicator(perfect=100, failed=0, pf_curve='linear', pf_interval=10)
        ci.sim(100)
        self.assertEqual(ci.get_condition(), 0)


    # ********** Test sim_timeline **********

    # early start

    def test_sim_timeline_early_start_early_stop(self):
        expected = np.concatenate((np.full(10,100), np.linspace(100,90,11)))
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear')
        
        cp = ci.sim_timeline(t_start=-10, t_stop=10)

        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_early_start_late_stop(self):
        expected = np.concatenate((np.full(10,100), np.linspace(100,50,51), np.full(50,50)))
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear')

        cp = ci.sim_timeline(t_start=-10, t_stop=100)

        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_early_start_no_stop(self):
        expected = np.concatenate((np.full(10,100), np.linspace(100,50,51)))
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear')
        
        cp = ci.sim_timeline(t_start=-10)

        np.testing.assert_array_equal(cp, expected)

    # existing condition

    def test_sim_timeline_existing_condition_early_start_early_stop(self):
        expected = np.concatenate((np.full(10,90), np.linspace(90,70,21)))
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear')

        ci.set_condition(90)
        cp = ci.sim_timeline(t_start=-10, t_stop = 20)

        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_existing_condition_early_start_late_stop(self):
        expected = np.concatenate((np.full(10,90), np.linspace(90,50,41), np.full(60,50)))
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear')

        ci.set_condition(90)
        cp = ci.sim_timeline(t_start=-10, t_stop = 100)

        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_existing_condition_early_start_no_stop(self):
        expected = np.concatenate((np.full(10,90), np.linspace(90,50,41)))
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear')

        ci.set_condition(90)
        cp = ci.sim_timeline(t_start=-10)

        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_existing_condition_v_early_start_v_late_stop(self):
        expected = np.concatenate((np.full(10, 90), np.linspace(90,50,41), np.full(60,50)))
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear') 

        ci.set_condition(90)
        cp = ci.sim_timeline(t_start=-10, t_stop=100)
        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_existing_condition_v_late_start_v_late_stop(self):
        expected = np.full(41,50)
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear') 

        ci.set_condition(50)
        cp = ci.sim_timeline(t_start=60, t_stop=100)

        np.testing.assert_array_equal(cp, expected)

    # late start

    def test_sim_timeline_late_start_early_stop(self):
        expected = np.linspace(95,90,6)
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear')
        
        cp = ci.sim_timeline(t_start=5, t_stop=10)

        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_late_start_late_stop(self):
        expected = np.concatenate((np.linspace(95,50,46), np.full(50, 50)))
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear')
        
        cp = ci.sim_timeline(t_start=5, t_stop=100)

        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_late_start_no_stop(self):
        expected = np.linspace(95,50,46)
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear')
        
        cp = ci.sim_timeline(t_start=5)

        np.testing.assert_array_equal(cp, expected)

    # no start

    def test_sim_timeline_no_start_early_stop(self):
        expected = np.linspace(100,90,11)
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear')
        
        cp = ci.sim_timeline(t_stop=10)

        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_no_start_late_stop(self):
        expected = np.concatenate((np.linspace(100,50,51), np.full(50,50)))
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear')
        
        cp = ci.sim_timeline( t_stop=100)

        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_no_start_no_stop(self):
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear')
        expected = np.linspace(100,50,51)
        cp = ci.sim_timeline()
        np.testing.assert_array_equal(cp, expected)        


    # very early start

    def test_sim_timeline_v_early_start_early_stop(self):
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear')
        expected = np.concatenate((np.full(100,100), np.linspace(100,90,11)))
        cp = ci.sim_timeline(t_start=-100, t_stop=10)
        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_v_early_start_late_stop(self):
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear')
        expected = np.concatenate((np.full(100,100), np.linspace(100,50,51), np.full(50,50)))
        cp = ci.sim_timeline(t_start=-100, t_stop=100)
        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_v_early_start_no_stop(self):
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear')
        expected = np.concatenate((np.full(100,100), np.linspace(100,50,51)))
        cp = ci.sim_timeline(t_start=-100)
        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_v_early_start_v_early_stop(self):
        expected = np.full(91,100)
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear')
        cp = ci.sim_timeline(t_start=-100, t_stop=-10)
        np.testing.assert_array_equal(cp, expected)

    # very late start

    def test_sim_timeline_v_late_start_early_stop(self):
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear')
        expected = [90]
        cp = ci.sim_timeline(t_start=20, t_stop=10)
        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_v_late_start_late_stop(self):
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear')
        expected = [50]
        cp = ci.sim_timeline(t_start=110, t_stop=100)
        np.testing.assert_array_equal(cp, expected)

    def test_sim_timeline_v_late_start_no_stop(self):
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear')
        expected = [50]
        cp = ci.sim_timeline(t_start=60)
        np.testing.assert_array_equal(cp, expected)


    # ********** Test set_condition **************** TODO add condition checks as well as time

    def test_set_condition_pf_decreasing_above_limit(self):
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear')

        ci.set_condition(150)

        self.assertEqual(ci.get_condition(), 100)
        self.assertEqual(ci.get_accumulated(), 0)


    def test_set_condition_pf_decreasing_below_limit(self):
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear')

        ci.set_condition(0)

        self.assertEqual(ci.get_accumulated(), 50)
        self.assertEqual(ci.get_condition(), 50)

    def test_set_condition_pf_decreasing_in_limit(self):
        ci = ConditionIndicator(perfect=100, failed=50,pf_interval=50, pf_curve='linear')

        ci.set_condition(70)

        self.assertEqual(ci.get_condition(), 70)
        self.assertEqual(ci.get_accumulated(), 30)


    def test_set_condition_pf_increasing_above_limit(self):
        ci = ConditionIndicator(perfect=50, failed=100,pf_interval=50, pf_curve='linear')

        ci.set_condition(150)

        self.assertEqual(ci.get_condition(), 100)
        self.assertEqual(ci.get_accumulated(), 50)


    def test_set_condition_pf_increasing_below_limit(self):
        ci = ConditionIndicator(perfect=50, failed=100,pf_interval=50, pf_curve='linear')

        ci.set_condition(0)

        self.assertEqual(ci.get_condition(), 50)
        self.assertEqual(ci.get_accumulated(), 0)


    def test_set_condition_pf_increasing_in_limit(self):
        ci = ConditionIndicator(perfect=50, failed=100,pf_interval=50, pf_curve='linear')

        ci.set_condition(70)

        self.assertEqual(ci.get_condition(), 70)
        self.assertEqual(ci.get_accumulated(), 20)


    # **************** Test limit_reached *****************

    """def test_is_failed_pf_decreasing_at_threshold (self):
        c = ConditionIndicator(100,0,'linear', [-10])
        c.threshold_failure = 50
        ci.set_condition(50)
        self.assertTrue(c.is_failed())

    def test_is_failed_pf_decreasing_exceeds_threshold (self):
        c = ConditionIndicator(100,0,'linear', [-10])
        c.threshold_failure = 50
        ci.set_condition(30)
        self.assertTrue(c.is_failed())

    def test_is_failed_pf_decreasing_within_threshold (self):
        c = ConditionIndicator(100,0,'linear', [-10])
        c.threshold_failure = 50
        ci.set_condition(70)
        self.assertFalse(c.is_failed())
    
    def test_is_failed_pf_increasing_at_threshold (self):
        c = ConditionIndicator(0,100,'linear', [10], decreasing = False)
        c.threshold_failure = 50
        ci.set_condition(50)
        self.assertTrue(c.is_failed())

    def test_is_failed_pf_increasing_exceeds_threshold (self):
        c = ConditionIndicator(0,100,'linear', [10], decreasing = False)
        c.threshold_failure = 50
        ci.set_condition(70)
        self.assertTrue(c.is_failed())

    def test_is_failed_pf_increasing_within_threshold (self):
        c = ConditionIndicator(0,100,'linear', [10], decreasing = False)
        c.threshold_failure = 50
        ci.set_condition(30)
        self.assertFalse(c.is_failed())"""

    # **************** Test the measuring functions

    def test_detectable_condition_starts_zero_current_above_detection_threshold_zero(self):
        ci = ConditionIndicator(perfect=0, failed=100, pf_curve='linear', pf_interval=10)
        ci.condition_detectable = 0
        ci.set_condition(30)
        self.assertTrue(ci.detectable())

    def test_detectable_condition_starts_zero_current_at_detection_threshold_zero(self):
        ci = ConditionIndicator(perfect=0, failed=100, pf_curve='linear', pf_interval=10)
        ci.condition_detectable = 0
        ci.set_condition(0)
        self.assertTrue(ci.detectable())

    def test_detectable_condition_starts_zero_current_above_detection_threshold_positive(self):
        ci = ConditionIndicator(perfect=0, failed=100, pf_curve='linear', pf_interval=10)
        ci.condition_detectable = 20
        ci.set_condition(30)
        self.assertTrue(ci.detectable())

    def test_detectable_condition_starts_zero_current_at_detection_threshold_positive(self):
        ci = ConditionIndicator(perfect=0, failed=100, pf_curve='linear', pf_interval=10)
        ci.condition_detectable = 20
        ci.set_condition(20)
        self.assertTrue(ci.detectable())

    def test_detectable_condition_starts_positive_current_below_detection_threshold_positive(self):
        ci = ConditionIndicator(perfect=100, failed=0, pf_curve='linear', pf_interval=10)
        ci.condition_detectable = 80
        ci.set_condition(ci.condition_perfect)
        self.assertFalse(ci.detectable())

    def test_detectable_condition_starts_positive_current_above_detection_threshold_zero(self):
        ci = ConditionIndicator(perfect=100, failed=0, pf_curve='linear', pf_interval=10)
        ci.condition_detectable = 80
        ci.set_condition(90)
        self.assertFalse(ci.detectable())

    def test_detectable_condition_starts_positive_current_at_detection_threshold_zero(self):
        ci = ConditionIndicator(perfect=100, failed=0, pf_curve='linear', pf_interval=10)
        ci.condition_detectable = 80
        ci.set_condition(0)
        self.assertTrue(ci.detectable())

    def test_detectable_condition_starts_positive_current_above_detection_threshold_positive(self):
        ci = ConditionIndicator(perfect=100, failed=0, pf_curve='linear', pf_interval=10)
        ci.condition_detectable = 80
        ci.set_condition(70)
        self.assertTrue(ci.detectable())

    def test_detectable_condition_starts_positive_current_at_detection_threshold_positive(self):
        ci = ConditionIndicator(perfect=100, failed=0, pf_curve='linear', pf_interval=10)
        ci.condition_detectable = 80
        ci.set_condition(80)
        self.assertTrue(ci.detectable())


    # **************** Test the reset functions ***********************

    def test_reset(self):
        ci = ConditionIndicator(perfect=100, failed=0, pf_curve='linear', pf_interval=10)
        ci.set_condition(50)
        ci.reset()
        self.assertEqual(ci.get_condition(),100)

    # **************** Test the reset_any functions ***********************
    
    def test_reset_any_reduction_factor_all(self):
        ci = ConditionIndicator(perfect=100, failed=0, pf_curve='linear', pf_interval=10)

        ci.set_condition(50)
        ci.reset_any(target=1, method='reduction_factor', axis='condition')
        condition = ci.get_condition()

        self.assertEqual(condition, 100)

    def test_reset_any_reduction_factor_half(self):
        ci = ConditionIndicator(perfect=100,failed=0,pf_curve='linear', pf_interval=10)
        ci.set_condition(50)
        ci.reset_any(target=0.5, method='reduction_factor', axis='condition')
        condition = ci.get_condition()

        self.assertEqual(condition, 75)

    def test_reset_any_reduction_factor_none(self):
        ci = ConditionIndicator(perfect=100,failed=0,pf_curve='linear', pf_interval=10)

        ci.set_condition(50)
        ci.reset_any(target=0, method='reduction_factor', axis='condition')
        condition = ci.get_condition()

        self.assertEqual(condition, 50)

    

    # ************** Test the accumulation functions ******************

    def test_accumulate_time(self):
        self.assertEqual(False, False)

if __name__ == '__main__':
    unittest.main()
