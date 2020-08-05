

import unittest

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

        fm = FailureMode(alpha=50, beta=1.5, gamma=10)
        fm.sim_timeline(200)
        fm.plot_timeline()


if __name__ == '__main__':
    unittest.main()