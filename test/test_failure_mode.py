


import unittest
import unittest.mock
import io

import utils

from pof.failure_mode import FailureMode                        
import pof.demo as demo

class TestFailureMode(unittest.TestCase):

    def setUp(self):
        # Failure distribtion
        self.alpha = 50
        self.beta = 1.5
        self.gamma = 10

        self.pf_interval = 5

        """self.fm = FailureMode(
            untreated = dict(alpha=self.alpha, beta=self.beta, gamma=self.gamma),
            pf_interval = self.pf_interval
        )"""

    def test_imports_correctly(self):
        self.assertTrue(True)

    def test_instantiate(self):
        try:
            fm = FailureMode()
            self.assertIsNotNone(fm)
        except ValueError:
            self.fail('ValueError returned')
        except:
            self.fail('Unknown error')

    def test_from_dict(self):
        try:
            fm = FailureMode.from_dict(demo.failure_mode_data['early_life'])
            self.assertIsNotNone(fm)
        except ValueError:
            self.fail('ValueError returned')
        except:
            self.fail('Unknown error')

    def test_from_dict_value_error_exits(self):

        false_data = dict(pf_curve = 'invalid_value')

        with self.assertRaises(ValueError):
            fm = FailureMode.from_dict(false_data)
            self.assertIsNotNone(fm)

    
    def test_instantiate_with_data(self):
        try:
            fm = FailureMode(name='random', untreated = dict(alpha=500, beta=1, gamma=0))
            self.assertIsNotNone(fm)
        except ValueError:
            self.fail('ValueError returned')
        except:
            self.fail('Unknown error')

    # ************ Test init_timeline ***********************

    def test_init_timeline_condition_step(self): #TODO full coverage
        t_start = 0
        t_end = 200
        fm = FailureMode().load(demo.failure_mode_data['random'])

        fm.init_timeline(t_start=0, t_end=200)

        # Check times match
        self.assertEqual(fm.timeline['time'][0], t_start, "First time does not equal t_start")
        self.assertEqual(fm.timeline['time'][-1], t_end, "Last time in timeline does not equal t_end")

        # Check states match
        self.assertEqual(fm.timeline['initiation'][0], fm.is_initiated(), "First initiation in timeline does not equal current initiation")
        self.assertEqual(fm.timeline['detection'][0], fm.is_detected(), "First detection in timeline does not equal current detection")
        self.assertEqual(fm.timeline['failure'][0], fm.is_failed(), "First Failure in timeline does not equal current failure")

        # Check conditions match
        # TODO move conditions to indicators first

        # Check tasks match
        #TODO rewrite time function in tasks first


    def test_init_timeline_condition_linear(self): # TODO full coverage
        t_start = 0
        t_end = 200
        fm = FailureMode.load(demo.failure_mode_data['slow_aging'])

        fm.init_timeline(t_start=0, t_end=200)

        # Check times match
        self.assertEqual(fm.timeline['time'][0], t_start, "First time does not equal t_start")
        self.assertEqual(fm.timeline['time'][-1], t_end, "Last time in timeline does not equal t_end")

        # Check states match
        self.assertEqual(fm.timeline['initiation'][0], fm.is_initiated(), "First initiation in timeline does not equal current initiation")
        self.assertEqual(fm.timeline['detection'][0], fm.is_detected(), "First detection in timeline does not equal current detection")
        self.assertEqual(fm.timeline['failure'][0], fm.is_failed(), "First Failure in timeline does not equal current failure")
        fm = FailureMode(demo.failure_mode_data['slow_aging'])

        # Check conditions match
        # TODO move conditions to indicators first copy from previous test

        # Check tasks match
        #TODO rewrite time function in tasks first copy from previous test


    # ************ Test sim_timeline ***********************

    def test_sim_timeline_condition_step(self): #TODO full coverage
        t_start = 0
        t_end = 200
        fm = FailureMode.load(demo.failure_mode_data['random'])

        initiation_start = fm.is_initiated()
        detection_start = fm.is_detected()
        failure_start = fm.is_failed()


        fm.sim_timeline(t_start=t_start, t_end=t_end)

        # Check times are ok
        self.assertEqual(fm.timeline['time'][0], t_start, "First time does not equal t_start")
        self.assertEqual(fm.timeline['time'][-1], t_end, "Last time in timeline does not equal t_end")

        # Check states are ok
        self.assertEqual(fm.timeline['initiation'][0], initiation_start, "First initiation in timeline does not equal current initiation")
        self.assertEqual(fm.timeline['initiation'][-1], fm.is_initiated(), "Last initiation in timeline does not equal current initiation")
        self.assertEqual(fm.timeline['detection'][0], detection_start, "First detection in timeline does not equal current detection")
        self.assertEqual(fm.timeline['detection'][-1], fm.is_detected(), "Last detection in timeline does not equal current detection")
        self.assertEqual(fm.timeline['failure'][0], failure_start, "First Failure in timeline does not equal current failure")
        self.assertEqual(fm.timeline['failure'][-1], fm.is_failed(), "Last Failure in timeline does not equal current failure")


        # Check conditions match
        # TODO move conditions to indicators first

        # Check tasks match
        #TODO rewrite time function in tasks first

    # ************ Test load ***********************
    

    def test_load_data_demo_data(self):
        try:
            fm = FailureMode.load(demo.failure_mode_data['slow_aging'])
            self.assertIsNotNone(fm)
        except ValueError:
            self.fail('ValueError returned')
        except:
            self.fail('Unknown error')



    def test_load_demo_no_data(self):
        try:
            fm = FailureMode.load()
            self.assertIsNotNone(fm)
        except ValueError:
            self.fail('ValueError returned')
        except:
            self.fail('Unknown error')



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

    def test_update(self):

        expected_list = [True]

        fm = FailureMode().set_demo()
        dash_ids = fm.get_dash_ids()

        for dash_id in dash_ids: #['FailureMode-fm-tasks-Task-inspection-trigger-condition-wall_thickness-lower']: #dash_ids:

            for expected in expected_list:
        
                fm.update(dash_id, expected)

                val = utils.get_dash_id_value(fm, dash_id)

                self.assertEqual(val, expected, msg = "Error: dash_id %s" %(dash_id))


    # ************ Test reset methods *****************

    # Change all condition, state and task count. Check values change or don't change for each of them

if __name__ == '__main__':
    unittest.main()