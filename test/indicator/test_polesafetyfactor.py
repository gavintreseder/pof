"""
    Filename: test_polesafetyfactor.py
    Description: Contains the code for testing the PoleSafetyFactor class
    Author:
        Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
"""

import unittest
from unittest.mock import Mock
import copy

import numpy as np

import fixtures
import testconfig  # pylint: disable=unused-import
from test_load import TestPofBase
from pof.indicator import PoleSafetyFactor
import pof.demo as demo



class TestPoleSafetyFactor(unittest.TestCase):

    def setUp(self):
        super().setUp()

        # TestInstantiate
        self._class = PoleSafetyFactor

        # TestPofBase
        self._data_valid = [dict(name="TestPoleSafetyFactor", pf_curve="pole_safety_factor")]
        self._data_invalid_values = [{"pf_curve": "invalid_value"}]
        self._data_invalid_types = [
            {"invalid_type": "invalid_type", "indicator_type": "ConditionIndicator"}
        ]
        self._data_complete = copy.deepcopy(fixtures.complete["pole_safey_factor"])

    def test_sim_timeline(self):
        NotImplemented


if __name__ == "__main__":
    unittest.main()
