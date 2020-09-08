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

    def test_load_no_data(self):
        task = Task().load()
        self.assertIsNotNone(task)

    def test_load_some_data(self):
        task = Task().load(demo.inspection_data['instant'])
        self.assertIsNotNone(task)

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

    def test_load_no_data(self):
        task = ConditionTask().load()
        self.assertIsNotNone(task)

    def test_load_some_data(self):
        task = ConditionTask().load(demo.on_condition_replacement_data)
        self.assertIsNotNone(task)

    def test_update(self):

        # Test all the options
        
        self.assertTrue(True)

class TestInspection(unittest.TestCase):

    def setUp(self):
        self.insp = Inspection(t_interval=5)
        self.c = Condition(perfect=100, limit=0, cond_profile_type = 'linear', cond_profile_params = [-10])

        # Counters
        self.n_sims = 1000
        self.n_detect = 0

    def test_instantiate(self):
        insp = Inspection()

        self.assertTrue(True)

    """Tests to complete


    inspection before conditions not met
    inspection with one condition met
    inspection with all conditions met

    """
    #def test_sim_inspect_with_p_detection_1_before_t_inspection(self):

        