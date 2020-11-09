import unittest

import scipy.stats as ss

import fixtures
import testconfig  # pylint: disable=unused-import
from test_load import TestPofBase
from pof.distribution import Distribution, DistributionManager


# class TestDistributionManager(unittest.TestCase):
#     def test_set(self):

#         dist = Distribution.demo()
#         dists = DistributionManager()

#         dists["untreated"] = dist

#     def test_update(self):
#         NotImplemented


class TestDistribution(TestPofBase, unittest.TestCase):
    def setUp(self):
        super().setUp()

        # TestPofBase
        self._class = Distribution
        self._data_valid = [{"name": "test"}]
        self._data_invalid_types = [{"alpha": "string"}]  # TODO[{"alpha": "string"}]
        self._data_invalid_values = [{"alpha": -1}]

        # Altartive method
        # self._data_invalid = [
        #     (Error, Input)
        #     (KeyError, {})
        # ]
        self._data_complete = [
            fixtures.complete["distribution_0"],
            fixtures.complete["distribution_1"],
        ]

        self.alpha = 50
        self.beta = 1.5
        self.gamma = 10

        self.dist = Distribution(alpha=50, beta=1.5, gamma=10)

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

        p_start = ss.weibull_min.sf(50, dist.beta, scale=dist.alpha, loc=dist.gamma)
        p_end = ss.weibull_min.sf(100, dist.beta, scale=dist.alpha, loc=dist.gamma)

        self.assertEqual(p_1[0], 1)
        self.assertEqual(p_2[0], 1)
        self.assertEqual(p_2[-1], p_end / p_start)

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

        p_start = ss.weibull_min.sf(50, dist.beta, scale=dist.alpha, loc=dist.gamma)
        p_end = ss.weibull_min.sf(100, dist.beta, scale=dist.alpha, loc=dist.gamma)

        self.assertEqual(p_1[0], 0)
        self.assertEqual(p_2[0], 0)
        self.assertEqual(p_2[-1], 1 - p_end / p_start)

        # Check the boundary cases

    def test_get_value(self):

        d = Distribution(alpha=10, beta=3, gamma=1)

        alpha = d.get_value(key="alpha")
        self.assertEqual(alpha, 10)

        alpha, beta, gamma = d.get_value(key=["alpha", "beta", "gamma"])
        self.assertEqual(alpha, 10)
        self.assertEqual(beta, 3)
        self.assertEqual(gamma, 1)

        # Create an object with parameters

        # use get value to get the value

        NotImplemented


if __name__ == "__main__":
    unittest.main()
