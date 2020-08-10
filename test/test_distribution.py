

# Add root folder to python path TODO figure out how to get rid of this
import sys, os
sys.path.append(os.path.dirname(os.getcwd()) + '/pof/')

import unittest

from pof.distribution import Distribution


class TestDistribution(unittest.TestCase):

    def setUp(self):
        self.alpha = 50
        self.beta = 1.5
        self.gamma = 10

        self.dist = Distribution(alpha=50, beta=1.5, gamma=10)

    def test_instantiate(self):
        dist = Distribution(alpha=50, beta=1.5, gamma=10)

        self.assertTrue(True)
        

    def test_all_tests_written(self):
        self.assertTrue(False)
        
    def test_likelihood(self):
        p_s1 = self.dist.likelihood(0, 10)
        p_s2 = self.dist.likelihood(10,50)

        self.assertLessEqual(p_s1, 0)
        self.assertGreaterEqual(p_s2, 0)

    # Check the boundary cases


if __name__ == '__main__':
    unittest.main()