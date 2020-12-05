"""
Check the test environemnt is going to work
"""
import unittest


class TestConfig(unittest.TestCase):

    def test_import_testconfig(self):
        import testconfig # pylint: disable=import-outside-toplevel

        self.assertIsNotNone(testconfig)

    def test_import_config(self):
        import config # pylint: disable=import-outside-toplevel

        self.assertIsNotNone(config)


if __name__ == "__main__":
    unittest.main()
