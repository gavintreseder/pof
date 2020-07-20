"""

Author: Gavin Treseder
"""

# ************ Packages ********************
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.linalg import circulant
from random import random, seed

from pof.degradation import Degradation
from pof.distribution import Distribution

#TODO move t somewhere else
#TODO create better constructors https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python

seed(1)

class Task:
    """
    Parameters:
                trigger
                    time, condition, symptom?, kpi?

                activty?
                    insp, repair, replace

    Things a task can do:
        - insp
            - detect symptom
            - measure condition
        - repair
            - stop initiation
            - improve condition
        - replace
            - reset everything

    """
    def __init__(self):
        self.activity = 'any'
        self.trigger = 'time'
        self.trigger_threshold_min = 0
        self.trigger_threshold_max = 10

        self.trigger_comp = '1.1.2'

        # TODO how the package is grouped together
        self.package = NotImplemented
        self._impacts_parent = NotImplemented
        self._impacts_children = False

        # Consumed per use
        self.cost = 100
        self.parts = 'pole' # TODO make this an object
        self.resources = 'trade'
        
        # Time to execute
        self.state = 'up' # or down

        # Log it's use
        self._count_checked = NotImplemented
        self._count_triggered = NotImplemented
        self._count_completed = NotImplemented

    def is_triggered(self):

        return


    def reset(self):
        self.triggered = False

    #TODO add methods for cost, resources and 



class Replace(Task): #TODO currenlty set up as emergency replacement

    def __init__(self):
        super().__init__()
        self.activity = 'replace' #TODO placeholder
        self.trigger = 'time' #TODO placeholder

    def check_trigger(self, parent):
        """
        """
        
        if parent._failed == True:
            self.triggered = True
        
        return self.triggered


class Inspection(Task):
    """
    Takes a condition (#TODO Symptoms) and determines if the failure has been detected
    """

    def __init__(self):
        super().__init__()

        self.t_last_inspection = 0
        self.t_inspection_interval = 5
        self.t_start_inspections = 0 #TODO add this feature
        self.p_detection = 0.9

    def inspect(self, degradation):
        """
        Check if the condition has been detected
        """

        #TODO rewrite this to include a measure method as opposed to simple detection

        if degradation.detectable() == True:

            if random() < self.p_detection:
                return True
        
        return False

    def sim_inspect(self, t_step, degradation):
        
        detected = False

        # Check if an inspection occurs in the simulation step
        if self.t_last_inspection <= self.t_inspection_interval and self.t_last_inspection + t_step >= self.t_inspection_interval:
            
            # Check if anything is detected
            if self.inspect(degradation) == True:
                detected = True

            self.t_last_inspection = self.t_last_inspection + t_step #TODO round it back to the inspection interval

        return detected

            
# TODO
"""
    inspection -> detect failure initiation
    repair -> remove failure initiation (Failure Modes that only reset)
    restore -> remove failure initaition and restore some condition
    replace -> reset component
"""