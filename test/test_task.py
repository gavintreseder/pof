import unittest
import copy

import utils

from pof.task import Task, ConditionTask, Inspection
import pof.demo as demo

import fixtures


class TestTask(unittest.TestCase):
    def test_imports_correctly(self):
        self.assertTrue(True)

    def test_instantiate(self):
        task = Task()
        self.assertIsNotNone(task)

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

        t = Task.from_dict()
        update = {"alpha": 10, "beta": 5}

        self.assertRaises(KeyError, t.update_from_dict, update)


class TestConditionTask(unittest.TestCase):
    def test_imports_correctly(self):
        self.assertTrue(True)

    def test_instantiate(self):
        task = ConditionTask()
        self.assertIsNotNone(task)

    # **************** test_load ***********************

    def test_load_empty(self):
        task = ConditionTask.load()
        self.assertIsNotNone(task)

    def test_load_valid_dict(self):
        task = ConditionTask.load(demo.on_condition_replacement_data)
        self.assertIsNotNone(task)

    def test_update(self):

        # Test all the options

        self.assertTrue(True)


class TestInspection(unittest.TestCase):
    def test_imports_correctly(self):
        self.assertTrue(True)

    def test_instantiate(self):
        task = Inspection()
        self.assertIsNotNone(task)

    # **************** test_load ***********************

    def test_load_empty(self):
        task = Inspection.load()
        self.assertIsNotNone(task)

    def test_load_valid_dict(self):
        task = Inspection.load(demo.inspection_data["instant"])
        self.assertIsNotNone(task)

    # **************** test_update ***********************

    def test_update(self):

        # Test all the options

        self.assertTrue(True)
