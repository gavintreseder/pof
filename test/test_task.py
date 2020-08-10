import unittest

import utils

from pof.task import Task, Inspection
from pof.condition import Condition

class TestTask(unittest.TestCase):

    def test_instantiate(self):
        task = Task()

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

        