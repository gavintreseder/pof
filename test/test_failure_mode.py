


import unittest
import unittest.mock
import io

import utils

from pof.failure_mode import FailureMode
from pof.distribution import Distribution
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

    def test_instantiate(self):
        fm = FailureMode()
        self.assertIsNotNone(fm)
    
    def test_instantiate_with_data(self):
        fm = FailureMode(name='random', untreated = dict(alpha=500, beta=1, gamma=0))
        self.assertIsNotNone(fm)

    def test_sim_timeline(self):

        fm = FailureMode(alpha=50, beta=1.5, gamma=0)
        fm.sim_timeline(200)

    # ************ Test load ***********************
    

    def test_load_data_demo_data(self):
        fm = FailureMode().load(demo.failure_mode_data['slow_aging'])
        self.assertIsNotNone(fm)

    """@unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def assert_stdout(self, n, expected_output, mock_stdout):
        fm = FailureMode().load(demo.failure_mode_data['slow_aging'])
        self.assertEqual(mock_stdout.getvalue(), expected_output)

    def test_load_data_demo_data_no_errors(self):
        self.assert_stdout(2, '1\n2\n')"""


    def test_load_demo_no_data(self):
        fm = FailureMode().load()
        self.assertIsNotNone(fm)



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

if __name__ == '__main__':
    unittest.main()