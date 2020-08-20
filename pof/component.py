"""

Author: Gavin Treseder
"""

# ************ Packages ********************
import numpy as np
import pandas as pd
import scipy.stats as ss

from failure_mode import FailureMode
from distribtuion import Distribution

#TODO create better constructors https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python

class Component():
    """
        Parameters:

        Methods:
            
    """

    def __init__(self):
        
        # Initial parameters
        self.age = 0
        self.age_last_insp = 0

        # Link to other componenets
        self._parent_id = NotImplemented
        self._children_id = NotImplemented
        
        # Parameter
        self.fm = dict()
        self.conditions = dict()

    def set_demo(self):

        self.fm = dict(
            fast_aging = FailureMode().set_default().set_failure_dist(Distribtuion(alpha=50, beta=2, gamma=20)),
            slow_aging = FailureMode().set_default().set_failure_dist(Distribution(alpha=100, beta=1.5, gamma=20)),
            random = FailureMode().set_default().set_failure_dist(Distribution(alpha=1000, beta=1, gamma=0)),
        )
    
        return self

    
    def sim_timeline(self, t_end, t_start=0):

        # Initialise the failure modes
        for fm in self.fm:
            fm.init_timeline(t_start=t_start, t_end=t_end)

        t_now = t_start

        while t_now < t_end:

            next_fm_tasks = self.next_tasks(t_now)

            self.complete_tasks(t_now, next_fm_tasks)

            t_now = t_now + 1

        self._sim_counter = self._sim_counter + 1
        self.reset_for_next_sim()


    def next_tasks(self, t_now):

        # TODO make this more efficent
        # TODO make this work if no tasks returned. Expect an error now

        # Get the task schedule for next tasks
        task_schedule = dict()
        for fm_name, fm in self.fm.items():
            t_now, task_names = fm.next_tasks(t_start=t_now)

            if t_now in task_schedule:
                task_schedule[t_now][fm_name] = task_names
            else:
                task_schedule[t_now] = dict(
                    fm_name = task_names,
                )

        t_now = min(task_schedule.keys())

        return task_schedule[t_now]

    def complete_tasks(self, t_now, fm_tasks):

        # TODO add logic around all the different ways of executing
        for fm_name, task_names in fm_tasks.items():
            self.fm[fm_name].complete_tasks(t_now, task_names)


    # Reset between other failure modes

    def load(self):
        # Load Failure Modes
        # Load asset information
        NotImplemented

    def reset_for_next_sim(self):

        for fm in self.fm.values():
            fm.reset_for_next_sim()

    def reset(self):

        NotImplemented


if __name__ == "__main__":
    component = Component()
    print("Component - Ok")