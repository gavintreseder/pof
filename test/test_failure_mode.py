

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
    
    def test_calc_prob_initiation_linear(self):
        
        fm = FailureMode(alpha=self.alpha, beta=self.beta, gamma=self.gamma)
        init_dist = Distribution(alpha=self.alpha, beta=self.beta, gamma=self.gamma - self.pf_interval)

        fm.calc_init_dist()

        self.assertEqual(fm.init_dist, init_dist)


    def test_get_probabilities_failure_not_initiated(self): # TODO change to prob thresholds rather than for weibull specifically
        self.fm.initiated = False
        
        dist = self.fm.init_dist

        # Age = 0
        p_i_s1 = self.fm.get_probabilities(dist.gamma)
        p_i_s2 = self.fm.get_probabilities(dist.alpha)

        # Age after failure can be iniitaited
        self.fm.age = self.fm.init_dist.gamma + 1
        p_i_s3 = self.fm.get_probabilities(dist.alpha)
        p_i_s4 = self.fm.get_probabilities(100000)

        self.assertLessEqual(p_i_s1, 0, "Initiation not possible during failure free period")
        self.assertGreaterEqual(p_i_s2, 0, "Initiation expected")
        self.assertGreaterEqual(p_i_s3, 0, "Initiation expected")
        self.assertGreaterEqual(p_i_s4, 0, "Initiation expected")

    def test_get_probabilities_failure_initiated(self):
        self.fm.initiated = True

        # Age = 0
        p_i_s1 = self.fm.get_probabilities(self.gamma)
        p_i_s2 = self.fm.get_probabilities(self.alpha)

        # Age after failure can be iniitaited
        self.fm.age = self.gamma + 1
        p_i_s3 = self.fm.get_probabilities(self.alpha)
        p_i_s4 = self.fm.get_probabilities(100000)

        self.assertEqual(p_i_s1, 1, "Failure already initiated")
        self.assertEqual(p_i_s2, 1, "Failure already initiated")
        self.assertEqual(p_i_s3, 1, "Failure already initiated")
        self.assertEqual(p_i_s4, 1, "Failure already initiated")

    def test_all_tests_written(self):
        self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()