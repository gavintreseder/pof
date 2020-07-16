

import unittest

from pof.degradation import Degradation

class TestDegradation(unittest.TestCase):

    def test_instantiate(self):
        deg = Degradation()

        self.assertTrue(True)
        

    def test_all_tests_written(self):
        self.assertTrue(False)
        

    # Check whole degradation
    # test_starts_perfect_ends_perfect
    # test_starts_perfect_ends_partial
    # test_starts_perfect_ends_failed
    # test_starts_partial_ends_partial
    # test_starts_partial_ends_partial
    # test_starts_partial_ends_failed

    # test_perfect_prior_to_start
    # test_partial_prior_to_start


if __name__ == '__main__':
    unittest.main()