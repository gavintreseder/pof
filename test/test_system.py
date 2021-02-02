import copy
import unittest
from unittest.mock import Mock, patch
import os

from test_pof_base import TestPofBaseCommon
from pof.paths import Paths
from config import config
from pof.system import System
import fixtures

cf = config["System"]


class TestSystem(TestPofBaseCommon, unittest.TestCase):
    """
    Unit tests for the System class including common tests from TestPoFBase
    """

    ## *************** Test setup ***********************

    def setUp(self):
        super().setUp()

        file_path = Paths().test_path + r"\fixtures.py"

        # TestPofBase Setup
        self._class = System
        self._data_valid = [dict(name="TestSystem")]
        self._data_invalid_types = [{"invalid_type": "invalid_type"}]
        self._data_invalid_values = []
        self._data_complete = copy.deepcopy(fixtures.complete["system"])

    def test_class_imports_correctly(self):
        self.assertIsNotNone(System)

    def test_class_instantiate(self):
        sys = System()
        self.assertIsNotNone(sys)

    # get_objects

    # get_dash_ids

    # get_update_ids

    # save

    ## *************** Test demo ***********************

    def test_demo(self):
        sys = System.demo()
        self.assertIsNotNone(sys)


if __name__ == "__main__":
    unittest.main()