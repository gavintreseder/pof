"""

Author: Gavin Treseder
"""

# ************ Packages ********************
import numpy as np
import pandas as pd
import scipy.stats as ss
from matplotlib import pyplot as plt
from random import random

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
    def __init__(self, perfect=100, limit=0):

        # Degradation details
        self.cond_type = 'loss' #TODO not used
        self.cond_profile_type = 'linear'
        self.pf_interval = 5

        # Time
        self.t_condition = 0
        self.t_max = 100 # TODO calculate this
        self.t_accumulated = 0

        # Condition
        self.condition_perfect = perfect
        self.condition_accumulated = 0
        self.condition = 0
        self.condition_threshold = 100
        self.condition_limit = limit

        # Methods 
        self.set_condition_profile()

        return
    
    def __str__(self):

        out = "Curve: %s\n" %(self.cond_profile_type)
        out = out + "PF Interval %s\n: " %(self.pf_interval)
        
        return out


    def set_condition(self, new_condition = None): # Rewrite

        if new_condition is None:
            self.condition = self.condition_perfect
        else:
            self.condition = new_condition
            self.t_condition = np.argmin(np.abs(self.condition_profile - new_condition))

        return

    def set_t_condition(self, t_condition):
        self.t_condition = t_condition

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

    # ************** Simulate Condition ***************

    def sim(self, t):
        """
        Increment the current time by t and return the new condition
        """
        self.t_condition = t + self.t_condition
        
        return self.current()

    # ************** Access Condition ******************

    def forecast(self, t):
        """
        Return the condition at a future time t
        """
        return self.at_time(t + self.t_condition)

    def current(self):
        """
        Return the condition at time t
        """
        return self.at_time(self.t_condition)

    def at_time(self, t):

        if t < 0:
            t = 0
        elif t >= len(self.condition_profile):
            t = len(self.condition_profile) - 1

        cond = self.condition_profile[t]

        return cond

    # *************** Measure Condition ****************

    def measure(self):
        """
        Returns a measurement of the condition based on uncertainty around its measurement
        """

        # TODO use variables
        # TODO add other methods
        # TODO make sure it works for positive and negative

        m_mean = self.current()
        m_sigma = (m_mean - self.condition_perfect)/6

        measurement = ss.norm.ppf(random(), loc=m_mean, scale=m_sigma)

        return measurement
    
    # *************** Reset **********************

    def reset_degradation(self, t_new=0, t_reverse=None, t_percent=None, cond_new=None, cond_reverse=None, cond_percent=None, var='time', method='reset'):
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

        cond = self.current()
        """if method == 'reset':
            cond = cond_new

        if method == 'rf':
            cond = self.current() * cond_percent

        if method == 'reverse':
            cond = self.current() - cond_reverse"""

        if self.condition_perfect < self.condition_limit:
            self.cond_accumulated = min(max(0, cond), self.condition_limit)
        else:
            self.cond_accumulated = min(max(0, cond), self.condition_perfect)

        self.t_condition=0

        return

    def plot_condition_profile(self):
        plt.plot(self.condition_profile)
        plt.plot(self.t_condition, self.current(), 'rd')
        plt.show()
        
