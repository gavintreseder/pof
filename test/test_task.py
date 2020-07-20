

import unittest

from pof.task import Task, Inspection
from pof.degradation import Degradation

class TestTask(unittest.TestCase):

    def test_instantiate(self):
        task = Task()

        self.assertTrue(True)



class TestInspection(unittest.TestCase):

    def setUp(self):
        self.insp = Inspection()
        self.d = Degradation(perfect=100, limit=0, cond_profile_type = 'linear', cond_profile_params = [-10])

        # Counters
        self.n_sims = 1000
        self.n_detect = 0

    def test_instantiate(self):
        insp = Inspection()

        self.assertTrue(True)

    def test_sim_inspect_with_p_detection_1_t_in_window(self):

        self.insp.p_detection = 1
        self.insp.t_last_inspection = 0
        self.insp.t_last_inspection = 5
        self.insp.t_inspection_interval = 5

        t_step = 2

        self.d.condition_detectable = 0
        
        # Check the time
        for i in range(0, self.n_sims):

            if self.insp.sim_inspect(t_step, self.d):
                
                self.n_detect = self.n_detect + 1
        
        p_detect = self.n_detect / self.n_sims

        self.assertAlmostEqual(p_detect, 1, places=2)

    #def test_sim_inspect_with_p_detection_1_before_t_inspection(self):

        