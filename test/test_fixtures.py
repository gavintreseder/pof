"""
Check the test fixtures are able to be imported
"""
import unittest


class TestFixtures(unittest.TestCase):
    def test_import_fixtures(self):
        """ Check the fixtures can be imported correctly"""
        import fixtures  # pylint: disable=import-outside-toplevel

        self.assertIsNotNone(fixtures)


if __name__ == "__main__":
    unittest.main()