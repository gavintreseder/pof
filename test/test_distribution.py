
import unittest

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
        
    def test_likelihood(self):
        p_s1 = self.dist.likelihood(0, 10)
        p_s2 = self.dist.likelihood(10,50)

        self.assertLessEqual(p_s1, 0)
        self.assertGreaterEqual(p_s2, 0)

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

                self.assertEqual(val, expected, msg = "Error: dash_id %s" %(dash_id))


if __name__ == '__main__':
    unittest.main()