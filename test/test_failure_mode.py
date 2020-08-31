


import unittest

import utils

from pof.failure_mode import FailureMode
from pof.distribution import Distribution

class TestFailureMode(unittest.TestCase):

    def setUp(self):
        # Failure distribtion
        self.alpha = 50
        self.beta = 1.5
        self.gamma = 10

        self.pf_interval = 5

        self.fm = FailureMode(alpha=self.alpha, beta=self.beta, gamma=self.gamma)

    def test_instantiate(self):
        fm = FailureMode(alpha=self.alpha, beta=self.beta, gamma=self.gamma)

        self.assertTrue(True)
    
    def test_sim_timeline(self):

        fm = FailureMode(alpha=50, beta=1.5, gamma=0)
        fm.sim_timeline(200)

    
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

        for dash_id in dash_ids:

            for expected in expected_list:
        
                fm.update(dash_id, expected)

                val = utils.get_dash_id_value(fm, dash_id)

                self.assertEqual(val, expected, msg = "Error: dash_id %s" %(dash_id))

if __name__ == '__main__':
    unittest.main()