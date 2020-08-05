"""

Author: Gavin Treseder
"""

# ************ Packages ********************
import numpy as np
import pandas as pd
import scipy.stats as ss

from pof.failure_mode import FailureMode

#TODO move t somewhere else
#TODO create better constructors https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python


class Component():
    """
        Parameters:

        Methods:
            
    """

    def __init__(self):
        
        self.age = 0
        self.age_last_insp = 0

        # Link to other componenets

        self._parent_id
        self._children_id
        

        return

    def demo(self):

        self.fms = dict(
            fast_aging = FailureMode(alpha=50, beta=2, gamma=20),
            slow_aging = FailureMode(alpha=100, beta=1.5, gamma=20),
            random = FailureMode(alpha=1000, beta=1, gamma=0)

        )
    
    def load(self):
        # Load Failure Modes
        # Load asset information
    

    def reset(self):
