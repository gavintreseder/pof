
import unittest
import scipy.stats as ss

import utils
from pof.distribution import Distribution




class TestDistribution(unittest.TestCase):

    def setUp(self):
        self.alpha = 50
        self.beta = 1.5
        self.gamma = 10

        self.dist = Distribution(alpha=50, beta=1.5, gamma=10)

    def test_instantiate(self):
        dist = Distribution()
        self.assertIsNotNone(dist)

    def test_csf_length(self):

        dist = Distribution(alpha=50, beta=1.5, gamma=10)

        p_1 = dist.csf(0, 0)
        p_2 = dist.csf(50, 100)

        self.assertEqual(len(p_1), 1)
        self.assertEqual(len(p_2), 51)

    def test_csf_start_and_end(self):

        dist = Distribution(alpha=50, beta=1.5, gamma=10)

        p_1 = dist.csf(0, 0)
        p_2 = dist.csf(50, 100)

        p_start = ss.weibull_min.sf(
            50, dist.beta, scale=dist.alpha, loc=dist.gamma)
        p_end = ss.weibull_min.sf(
            100, dist.beta, scale=dist.alpha, loc=dist.gamma)

        self.assertEqual(p_1[0], 1)
        self.assertEqual(p_2[0], 1)
        self.assertEqual(p_2[-1], p_end/p_start)

    def test_cff_length(self):

        dist = Distribution(alpha=50, beta=1.5, gamma=10)

        p_1 = dist.cff(0, 0)
        p_2 = dist.cff(50, 100)

        self.assertEqual(len(p_1), 1)
        self.assertEqual(len(p_2), 51)

    def test_cff_start_and_end(self):

        dist = Distribution(alpha=50, beta=1.5, gamma=10)

        p_1 = dist.cff(0, 0)
        p_2 = dist.cff(50, 100)

        p_start = ss.weibull_min.sf(
            50, dist.beta, scale=dist.alpha, loc=dist.gamma)
        p_end = ss.weibull_min.sf(
            100, dist.beta, scale=dist.alpha, loc=dist.gamma)

        self.assertEqual(p_1[0], 0)
        self.assertEqual(p_2[0], 0)
        self.assertEqual(p_2[-1], 1 - p_end/p_start)

        # Check the boundary cases

        # ************ Test update methods *****************

    def test_update(self):

        expected_list = [True, False, 10, 'abc']

        dist = Distribution(alpha=50, beta=1.5, gamma=10)
        dash_ids = dist.get_dash_ids()

        for dash_id in dash_ids:

            for expected in expected_list:

                dist.update(dash_id, expected)

                val = utils.get_dash_id_value(dist, dash_id)

                self.assertEqual(
                    val, expected, msg="Error: dash_id %s" % (dash_id))



    def test_get_value(self):

        # Create an object with parameters

        # use get value to get the value
        
        #check those values are the same

        NotImplemented
    
    def test_update(self)

        #Create an ojbect

        # set a valuue

        # get that value

        # Check they match

        NotImplemented

if __name__ == '__main__':
    unittest.main()
