"""

Author: Gavin Treseder
"""

# ************ Packages ********************
import numpy as np
import pandas as pd
import scipy.stats as ss
from matplotlib import pyplot as plt
from random import random

from pof.condition import Condition

#TODO move t somewhere else
#TODO create better constructors https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python

# Overloadthis for timber deay

# Simple _Sytmpom

class ConditionContainer(Condition): 

    """
    Parameters: pf_curve : str
                    step, exponential, linear, normal
                
                pf_curve

                pf_interval : float

                p_detection : float
                    Probability that the 

                increasing : bool
                    Either increasing or decreasing
                
    """
    def __init__(self, dict_conditions):

        self.dict_conditions = dict_conditions


    def validate(self):
        """
        Check the parameters provided are valid
        """
        for 

        return True

    def set_condition(self, new_condition = None): # Rewrite

        if new_condition is None:
            self.condition = self.condition_perfect
        else:
            self.condition = new_condition
            
            self.t_condition = np.argmin(np.abs(self.condition_profile - new_condition))

        return

    def set_t_condition(self, t_condition): # TODO test time is within limits
        if t_condition < 0:
            self.t_condtiion = 0
        elif t_condition > self.t_max:
            self.t_condition = self.t_max
        else:
            self.t_condition = t_condition

    def set_condition_profile(self, t_min=0, t_max=100): # Change limits for time to match max degradation profile
        """
        Sets a condition profile based on the input parameters
        """

        # Calculte max t TODO
        x = np.linspace(0, t_max - t_min, t_max - t_min + 1)

        # Change to another function swtich type TODO https://stackoverflow.com/questions/60208/replacements-for-switch-statement-in-python 

        # Get the condition profile
        if self.pf_curve == 'linear':
            m = self.pf_curve_params[0]
            b = self.condition_perfect

            y = m * x + b

        elif self.pf_curve == 'step':
            y = NotImplemented


        # Add the accumulated time
        cond_lost = y[self.t_accumulated] - self.condition_perfect
        y = y + cond_lost
        y = y[self.t_accumulated:]

        # Add the accumulated condition
        if self.decreasing:

            y = y - self.condition_accumulated
            self.t_max = np.argmax(y <= self.threshold_failure)
            y = y[:self.t_max + 1]
            y[y < self.threshold_failure] = self.threshold_failure

        else:
            y = y + self.condition_accumulated
            self.t_max = np.argmax(y >= self.threshold_failure)
            y = y[:self.t_max + 1]
            y[y > self.threshold_failure] = self.threshold_failure


        self.condition_profile = y

    # ************** Simulate Condition ***************

    def sim(self, t):
        """
        Increment the current time by t and return the new condition
        """
        self.t_condition = min(t + self.t_condition,self.t_max)
        
        return self.current()

    # ************** Check Failure *********************

    def is_failed(self):
        
        if self.decreasing == True and self.condition_profile[self.t_condition] <= self.threshold_failure:
            return True
        
        if self.decreasing == False and self.condition_profile[self.t_condition] >= self.threshold_failure:
            return True

        return False

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

    # **************** Event series **********

    def get_condition_profile(self, t_stop = None, t_start = 0): #TODO this probably needs a delay?
        """
        
        """
        
        if t_stop == None:
            t_stop = self.t_max

        cp = self.condition_profile[np.arange(0, min(t_stop, self.t_max) + 1, 1)]

        # Fill the start with the current condition
        if t_start > 0:
            cp = np.append(np.full(t_start, self.condition), cp)
        
        # Fill the end with the failed condition
        if t_stop - t_start > self.t_max:
            cp = np.append(cp, np.full(t_stop - t_start - self.t_max, self.condition_failed))

        return cp

    def check_failure_event(self, t_stop = None, t_start = 0): #TODO this probably needs a delay? and can combine with condtion profile to make it simpler
        """
        Return the failure event series for a condition
        """

        cp = self.get_condition_profile(t_stop, t_start)

        if self.decreasing == True:
            e_f = (cp <= self.threshold_failure)
        else:
            e_f = (cp >= self.threshold_failure)

        return cp


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
    
    def is_detectable(self):
        #TODO rewrite to include the measurement

        if self.decreasing == True:

            if self.current() <= self.threshold_detection:
                return True
            
        if self.decreasing == False:

            if self.current() >= self.threshold_detection:
                return True

        return False

    # *************** Reset **********************

    def reset(self):
        
        # Reset the time
        self.t_condition = 0
        self.t_accumulated = 0
        self.t_max = None

        # Reset the condition
        self.condition = self.condition_perfect
        self.condition_accumulated = 0

        # Reset the condition profile 
        self.set_condition_profile()

    def restore(self):

        return 0


    def reset_condition(self, target=0, reduction_factor=1, reverse=0, method='reset', axis='time'):
        """
        # TODO make this work for all the renewal processes (as-bad-as-old, as-good-as-new, better-than-old, grp)
        """

        new = target
        current = self.current()

        if method == 'reduction_factor':
            new = current * reduction_factor

        elif method == 'reverse':
            new = current - reverse

        # Calculate the accumulated condition
        if axis == 'time':

            self.t_accumulated = int(min(max(0, new), self.t_max))

        elif axis == 'condition':

            if self.decreasing:
                self.condition_accumulated = min(max(self.condition_failed, new), self.condition_perfect)
            else:
                self.condition_accumulated = max(min(self.condition_failed, new), self.condition_perfect)

        self.set_condition_profile()
        
        return

    def plot_condition_profile(self):
        plt.plot(self.condition_profile)
        plt.plot(self.t_condition, self.current(), 'rd')
        plt.show()
        

class Symptom():

    # TODO consider combining symptom and degradation?
    def __init__(self):
        
        self.time_to_failure = 5 #TODO update


