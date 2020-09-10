import unittest

import utils

from pof.task import Task, ConditionTask, Inspection
from pof.condition import Condition
import pof.demo as demo

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
        task = Task.load(demo.inspection_data['instant'])
        self.assertIsNotNone(task)

    # **************** test_update ***********************

    def test_update(self):

        # Test all the options
        
        self.assertTrue(True)


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
        task = Inspection.load(demo.inspection_data['instant'])
        self.assertIsNotNone(task)

    # **************** test_update ***********************

    def test_update(self):

        # Test all the options
        
        self.assertTrue(True)