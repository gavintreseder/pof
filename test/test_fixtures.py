"""
Check the test environemnt is going to work
"""
import unittest

class TestFixtures(unittest.TestCase):

    def test_import(self):
        import fixtures
        self.assertIsNotNone(fixtures)


class TestTestConfig(unittest.TestCase):

    def test_import(self):
        import testconfig
        self.assertIsNotNone(testconfig)