

import unittest

from pof.distribution import Distribution


class TestDistribution(unittest.TestCase):

    def test_class_creation(self):
        dist = Distribution(alpha = 50, beta = 1.5, gamma = 10)

        self.assertTrue(True)
        

    def test_degradation_array(self):
        self.assertTrue(False)
        

    # Check the boundary cases


if __name__ == '__main__':
    unittest.main()