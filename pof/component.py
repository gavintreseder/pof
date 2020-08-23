"""

Author: Gavin Treseder
"""

# ************ Packages ********************
import numpy as np
import pandas as pd
import scipy.stats as ss

from tqdm import tqdm
from lifelines import WeibullFitter

from failure_mode import FailureMode
from distribution import Distribution

#TODO create better constructors https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python
#TODO create get, set, del and add methods


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

        # Simulation traking
        self._sim_counter = 0
        self._timelines = dict()

    # ****************** Load data ******************

        def load(self):
        # Load Failure Modes
        # Load asset information
        NotImplemented


    # ****************** Timeline ******************

    def mc_timeline(self, t_end, t_start=0, n_iterations=100):

        self.reset()  # TODO ditch this

        for i in tqdm(range(n_iterations)):
            self._timelines[i] = self.sim_timeline(t_end=t_end, t_start=t_start)
    
    def sim_timeline(self, t_end, t_start=0):

        # Initialise the failure modes
        self.init_timeline(t_start=t_start, t_end = t_end)

        t_now = t_start

        while t_now < t_end:

            t_next, next_fm_tasks = self.next_tasks(t_now)

            self.complete_tasks(t_next, next_fm_tasks)

            t_now = t_next + 1

        self._sim_counter = self._sim_counter + 1
        self.reset_for_next_sim()

    def init_timeline(self, t_end, t_start=0):
        """ Initilialise the timeline"""
        for fm in self.fm.values():
            fm.init_timeline(t_start=t_start, t_end=t_end)

    def next_tasks(self, t_now):
        """
        Returns a dictionary with the failure mode triggered
        """
        # TODO make this more efficent
        # TODO make this work if no tasks returned. Expect an error now

        # Get the task schedule for next tasks
        task_schedule = dict()
        for fm_name, fm in self.fm.items():

            t_now, task_names = fm.next_tasks(t_start=t_now)

            if t_now in task_schedule:
                task_schedule[t_now][fm_name] = task_names
            else:
                task_schedule[t_now] = dict()
                task_schedule[t_now][fm_name] = task_names

        t_now = min(task_schedule.keys())

        return t_now, task_schedule[t_now]

    def complete_tasks(self, t_now, fm_tasks):
        """Complete any tasks in the dictionary fm_tasks at t_now"""

        # TODO add logic around all the different ways of executing
        # TODO check task groups
        # TODO check value?
        # TODO add task impacts
        for fm_name, task_names in fm_tasks.items():
            self.fm[fm_name].complete_tasks(t_now, task_names)

    # ****************** Expected ******************

    # PoF

    def expected_pof(self):
        
        # 

    # RUL

    # Cost

    # ****************** Reset ******************

    def reset_for_next_sim(self):
        """ Reset parameters back to the initial state"""

        for fm in self.fm.values():
            fm.reset_for_next_sim()


    def reset(self):
        """ Reset all parameters back to the initial state and reset sim parameters"""

        # Reset failure modes
        for fm in self.fm.values():
            fm.reset()

        # Reset timelines
        self._timelines = dict()

        # Reset counters
        self._sim_counter = 0

    # ****************** Demonstration parameters ******************

    def set_demo(self):
        """ Loads a demonstration data set if no parameters have been set already"""

        if not self.fm
            self.fm = dict(
                fast_aging = FailureMode(alpha=50, beta=2, gamma=20).set_demo(),
                slow_aging = FailureMode(alpha=100, beta=1.5, gamma=20).set_demo(),
                random = FailureMode(alpha=1000, beta=1, gamma=0).set_demo(),
            )
    
        return self

    def reset_demo(self):
        """ Loads a demonstration data set for all parameters"""
        self.fm = dict()

        self.set_demo()


if __name__ == "__main__":
    component = Component()
    print("Component - Ok")