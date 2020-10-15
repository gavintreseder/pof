import unittest
from unittest.mock import Mock, MagicMock, patch
import copy

import utils

from pof.task import Task, ScheduledTask, ConditionTask, Inspection
import pof.demo as demo

import fixtures


class TestCommon(unittest.TestCase):
    def setUp(self):
        self._class = Mock(return_value=None)

    def test_class_imports_correctly(self):
        self.assertIsNotNone(self._class)

    def test_class_instantiate(self):
        task = self._class()
        self.assertIsNotNone(task)


class TestTask(make_test_case(Task)):
    def setUp(self):
        super().setUp()
        self._class = Task

    # **************** test_load ***********************

    def test_load_empty(self):
        task = Task.load()
        self.assertIsNotNone(task)

    def test_load_valid_dict(self):
        task = Task.load(demo.inspection_data["instant"])
        self.assertIsNotNone(task)

    # **************** test_update ***********************

    def test_update(self):

        test_data_1 = copy.deepcopy(fixtures.on_condition_replacement_data)
        test_data_1["cost"] = 0
        test_data_1["triggers"]["condition"]["fast_degrading"]["upper"] = 90
        test_data_2 = copy.deepcopy(fixtures.on_condition_replacement_data)

        # Test all the options
        t1 = Task.from_dict(test_data_1)
        t2 = Task.from_dict(test_data_2)

        t1.update_from_dict(
            {"cost": 5000, "trigger": {"condition": {"fast_degrading": {"upper": 20}}}}
        )

        # self.assertEqual(t1.__dict__, t2.__dict__)
        self.assertEqual(t1.cost, t2.cost)
        self.assertEqual(t1.triggers, t2.triggers)

    def test_update_error(self):

        test_data = copy.deepcopy(fixtures.on_condition_replacement_data)

        t = Task.from_dict(test_data)

        update = {"alpha": 10, "beta": 5}

        self.assertRaises(KeyError, t.update_from_dict, update)


class TestScheduledTask(make_test_case(ScheduledTask)):
    def setUp(self):
        #super().setUp()
        #self._class = ScheduledTask
        pass


del TestCommon

if __name__ == "__main__":
    unittest.main()