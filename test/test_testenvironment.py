"""
Check the test environemnt is going to work
"""
import unittest


class TestTestEnvironment(unittest.TestCase):
    def test_import_fixutres(self):
        import fixtures

        self.assertIsNotNone(fixtures)

    def test_import_testconfig(self):
        import testconfig

        self.assertIsNotNone(testconfig)
