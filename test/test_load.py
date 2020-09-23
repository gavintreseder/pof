"""
    Filename: test_load.py
    Description: Contains the code for testing the Constructor class
    Author: Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
 
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import copy

import utils

from pof.load import Load
import pof.demo as demo
import fixtures

from pof.task import Task, ScheduledTask, ConditionTask, Inspection
from pof.indicator import ConditionIndicator
from pof.distribution import Distribution
from pof.failure_mode import FailureMode

"""
class TestLoad(unittest.TestCase):
    @patch(cf.on_error_use_default, False)
    def test_first_case(self):
        self.assertRaises(Load.load, "Invalid Data")

    @patch(cf.on_error_use_default, False)
    def test_n(self):
        ldr = Load()
        ldr = Load("Invalid_data")
        self.assert"""


class TestLoad(unittest.TestCase):
    def test_update_failure_mode(self):

        test_data_1_fix = fixtures.failure_mode_data["early_life"]
        test_data_2_fix = fixtures.failure_mode_data["random"]

        fm1 = FailureMode.from_dict(test_data_1_fix)
        fm2 = FailureMode.from_dict(test_data_2_fix)

        fm1.update_from_dict(test_data_2_fix)

        self.assertEqual(fm1, fm2)

    def test_update_distribution(self):

        test_data_1 = {"alpha": 5, "beta": 3, "gamma": 1}
        test_data_2 = {"alpha": 10, "beta": 5, "gamma": 1}

        d1 = Distribution.from_dict(test_data_1)
        d2 = Distribution.from_dict(test_data_2)

        d1.update({"alpha": 10, "beta": 5})

        self.assertEqual(d1, d2)

    def test_update_indicator(self):

        test_data_1 = copy.deepcopy(fixtures.condition_data["fast_degrading"])
        test_data_1["name"] = "FD"
        test_data_1["pf_std"] = 0.25
        test_data_2 = copy.deepcopy(fixtures.condition_data["fast_degrading"])

        c1 = ConditionIndicator.from_dict(test_data_1)
        c2 = ConditionIndicator.from_dict(test_data_2)

        c1.update({"name": "fast_degrading", "pf_std": 0.5})

        self.assertEqual(c1, c2)

    def test_update_error_indicator(self):

        test_data = copy.deepcopy(fixtures.condition_data["fast_degrading"])

        c = ConditionIndicator.from_dict(test_data)
        update = {"alpha": 10, "beta": 5}

        self.assertRaises(KeyError, c.update, update)

    def test_update_task(self):
        # TODO: Once task dataclass change update_from_dict to update and self.assertEqual(t1, t2)
        test_data_1 = copy.deepcopy(fixtures.inspection_data["instant"])
        test_data_1["cost"] = 0
        test_data_1["triggers"]["condition"]["instant"]["upper"] = 90
        test_data_2 = copy.deepcopy(fixtures.inspection_data["instant"])

        t1 = Inspection.from_dict(test_data_1)
        t2 = Inspection.from_dict(test_data_2)

        t1.update_from_dict(
            {
                "cost": 50,
                "trigger": {"condition": {"instant": {"upper": 0}}},
            }
        )

        # self.assertEqual(t1, t2)
        self.assertEqual(t1.cost, t2.cost)
        self.assertEqual(t1.triggers, t2.triggers)

    def test_update_error_task(self):
        # TODO: Once task dataclass change update_from_dict to update
        test_data = copy.deepcopy(fixtures.inspection_data["instant"])

        t = Inspection.from_dict(test_data)

        update = {"alpha": 10, "beta": 5}

        self.assertRaises(KeyError, t.update_from_dict, update)
