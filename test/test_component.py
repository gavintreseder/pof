
"""
    Filename: test_component.py
    Description: Contains the code for testing the Component class
    Author: Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
 
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

import utils

from pof.component import Component

class TestComponent(unittest.TestCase):

    def setUp(self):
        
        comp = Component().set_demo()

    def test_class_imports_correctly(self):
        self.assertTrue(True)

    def test_class_instantiate(self):
        comp = Component()
        self.assertTrue(True)

    # *************** Test init_timeline ***********************

    def test_init_timeline(self):
        t_end = 200
        comp = Component().set_demo()
        comp.init_timeline(t_end)

        for fm in comp.fm.values():
            t_fm_timeline_end = fm.timeline['time'][-1]

            self.assertEqual(t_end, t_fm_timeline_end)

    # *************** Test complete_tasks ***********************

    def test_complete_tasks_one_fm_one_task(self):
        fm_next_tasks = dict(
            slow_aging = ['inspection'],
        )
        t_now = 5
        comp = Component().set_demo()
        comp.init_timeline(200)
        comp.complete_tasks(t_now, fm_next_tasks)
        
        for fm_name, fm in comp.fm.items():
            for task_name, task in fm.tasks.items():

                if fm_name in list(fm_next_tasks):
                    if task_name in fm_next_tasks[fm_name]:
                        self.assertEqual([t_now], task.t_completion)
                    else:
                        self.assertEqual([], task.t_completion)
                else:
                    self.assertEqual([], task.t_completion)

    def test_complete_tasks_two_fm_two_task(self):
        fm_next_tasks = dict(
            slow_aging = ['inspection', 'cm'],
            fast_aging = ['inspection', 'cm'],
        )
        t_now = 5
        comp = Component().set_demo()
        comp.init_timeline(200)
        comp.complete_tasks(t_now, fm_next_tasks)
        
        for fm_name, fm in comp.fm.items():
            for task_name, task in fm.tasks.items():

                if fm_name in list(fm_next_tasks):
                    if task_name in fm_next_tasks[fm_name]:
                        self.assertEqual([t_now], task.t_completion)
                    else:
                        self.assertEqual([], task.t_completion)
                else:
                    self.assertEqual([], task.t_completion)

    # *************** Test next_tasks ***********************

    def test_next_tasks_one_fm_one_task(self):

        t_now = None
        test_next_task = dict(
            slow_aging = (5, ['inspection']),
            fast_aging = (10, ['inspection', 'cm']),
            random = (15, ['inspection'])
        )

        expected = {k: v[1] for k, v in test_next_task.items() if v[0] == 5}

        comp = Component().set_demo()

        for fm_name, fm in comp.fm.items():
            fm.next_tasks = MagicMock(return_value = test_next_task[fm_name])
        
        t_next, next_task = comp.next_tasks(t_now)
        
        self.assertEqual(next_task, expected)
        self.assertEqual(t_next, 5)


    def test_next_tasks_many_fm_many_task(self):

        times = dict(
            slow_aging= [5,5,5],
            fast_aging = [10,5,5],
            random = [10,10,5],
        )

        for i in range(3):
            t_now = None
            test_next_task = dict(
                slow_aging = (times['slow_aging'][i], ['inspection']),
                fast_aging = (times['fast_aging'][i], ['inspection', 'cm']),
                random = (times['random'][i], ['inspection'])
            )

            expected = {k: v[1] for k, v in test_next_task.items() if v[0] == 5}

            comp = Component().set_demo()

            for fm_name, fm in comp.fm.items():
                fm.next_tasks = MagicMock(return_value = test_next_task[fm_name])
            
            t_next, next_task = comp.next_tasks(t_now)
            
            self.assertEqual(next_task, expected)
            self.assertEqual(t_next, 5)

    # *************** Test sim_timeline ***********************

    def test_sim_timeline_active_all(self):
        comp = Component().set_demo()

        comp.sim_timeline(200)
        print(comp)

    def test_sim_timline_active_one(self):
        comp = Component().set_demo()
        
        comp.fm[list(comp.fm)[0]].active=False
        comp.sim_timeline(200)
        print(comp)


    # ************ Test update methods *****************

    def test_update(self):

        expected_list = [True]

        comp = Component().set_demo()
        dash_ids = comp.get_dash_ids()

        for dash_id in dash_ids:

            for expected in expected_list:
        
                comp.update(dash_id, expected)

                val = utils.get_dash_id_value(comp, dash_id)

                self.assertEqual(val, expected, msg = "Error: dash_id %s" %(dash_id))

if __name__ == '__main__':
    unittest.main()
