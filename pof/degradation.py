"""

Author: Gavin Treseder
"""

# ************ Packages ********************
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.linalg import circulant

from pof.distribution import Distribution

#TODO move t somewhere else
#TODO create better constructors https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python


class Degradation():

    """
    Parameters: pf_curve : str
                    step, exponential, linear, normal
                
                pf_interval : float

                p_detection : float
                    Probability that the 
                
    """
    def __init__(self, perfect=0, limit=100):

        self.cond_type = 'loss' #TODO not used
        self.cond_profile_type = 'linear'
        self.pf_interval = 5

        self.condition_perfect = perfect
        self.condition_accumulated = 0
        self.condition = 0
        self.condition_threshold = 100
        self.condition_limit = limit

        self.set_condition_profile()

        self.t_condition = 0
        self.t_max = 100 # TODO calculate this

        self.t_accumulated = 0

        return
    
    def __str__(self):

        out = "Curve: %s\n" %(self.cond_profile_type)
        out = out + "PF Interval %s\n: " %(self.pf_interval)
        
        return out


    def set_condition(self, new_condition = None): # Rewrite

        if new_condition is None:
            self.condition = self.condition_limit
        else:
            self.condition = new_condition

        return

    def set_condition_profile(self, t_min=0, t_max=100): # Change limits for time to match max degradation profile
        """
        Sets a condition profile based on the input parameters
        """

        x = np.linspace(0, t_max - t_min, t_max - t_min + 1)

        # Change to another function swtich type TODO https://stackoverflow.com/questions/60208/replacements-for-switch-statement-in-python 

        # Get the condition profile
        if self.cond_profile_type == 'linear':
            m = -10
            b = self.condition_perfect

            y = m * x + b

        # Add the accumulated time
        cond_lost = y[self.t_accumulated] - self.condition_perfect
        y = y + cond_lost
        y = y[self.t_accumulated:]

        # Add the accumulated condition
        if self.condition_perfect < self.condition_limit:
            y = y + self.condition_accumulated
            y[y > self.condition_limit] = self.condition_limit
        else:
            y = y - self.condition_accumulated
            y[y < self.condition_limit] = self.condition_limit


        self.condition_profile = y


    def expected(self, t = None, direction = 'loss'):

        if t < 0:
            t = 0
        elif t > len(self.condition_profile):
            t = len(self.condition_profile)

        cond = self.condition_profile[t]

        return cond

    def expected_range(self, t=None,):
        # Placeholder for when this is updated to a range
        return

    def reset_degradation(self, t_new=None, t_reverse=None, t_percent=None, cond_new=None, cond_reverse=None, cond_percent=None, var='time', method='reset'):
        """
        # TODO make this work for all the renewal processes (as-bad-as-old, as-good-as-new, better-than-old, grp)
        """
        
        # Calculated accumulated time
        if method == 'reset':
            t = t_new

        if method == 'rf':
            t = self.t_condition * t_percent

        if method == 'reverse':
            t = self.t_condition - t_reverse

        self.t_accumulated = int(min(max(0, t), self.t_max))

        # Calculate accumulated condition 
        cond_now = self.condition_profile[self.t_condition]
        if method == 'reset':
            cond = cond_new

        if method == 'rf':
            cond = cond_now * cond_percent

        if method == 'reverse':
            cond = cond_now - cond_reverse

        if self.condition_perfect < self.condition_limit:
            self.cond_accumulated = min(max(0, cond), self.condition_limit)
        else:
            self.cond_accumulated = min(max(0, cond), self.condition_perfect)

        return
        
