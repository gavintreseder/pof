"""

Author: Gavin Treseder
"""

# ************ Packages ********************
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.linalg import circulant
from random import random
from random import seed

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
        self.package
        self._impacts_parent
        self._impacts_children = False

        # Consumed per use
        self.cost = 100
        self.parts = 'pole' # TODO make this an object
        self.resources = 'trade'
        
        # Time to execute
        self.state = 'up' # or down

        # Log it's use
        self._count_checked
        self._count_triggered
        self._count_completed

    def is_triggered(self):

        return


    def reset(self):
        self.triggered = False

    #TODO add methods for cost, resources and 


class Replace(Task): #TODO currenlty set up as emergency replacement

    def __init__(self):
        super().__init__()
        self.activity = 'replace'
        self.trigger = 

    def check_trigger(self, parent):
        """
        """
        
        if parent._failed == True
            self.triggered = True
        
        return self.triggered

    